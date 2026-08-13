import hashlib
import os
from pathlib import Path
import pandas as pd
import re
from typing import Dict
import csv

STRUCTURED_PII_COLS = [
    "NAME", "EMAIL", "NUMERO_TELEFONO", "RUC_CEDULA",
    "USUARIO_CREADORI", "CUST_ACCT", "CUST_ACCT_NUMBER",
    "CTA_FACTURACION", "CLIENTE_ID", "DIRECCALLE",
    "NOMB_ASIG", "NOMB_CIERR","NOM_USUARIO"
]

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DIR_PATH = (BASE_DIR / "db_files").resolve()

def anonymize_column(series: pd.Series, prefix: str) -> pd.Series:
    """Reemplaza valores reales con IDs sintéticos determinísticos."""
    mapping = {}
    for val in series.dropna().unique():
        # Hash determinístico para mantener joins consistentes
        h = hashlib.sha256(str(val).encode()).hexdigest()[:8]
        mapping[val] = f"{prefix}_{h}"
    return series.map(mapping)


def _stable_token(value: object, prefix: str) -> str:
    s = str(value).strip()
    h = hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}_{h}"


_CEDULA_COEFFS = [2, 1, 2, 1, 2, 1, 2, 1, 2]


def _is_valid_cedula_ec(digits: str) -> bool:
    """Valida cédula ecuatoriana (algoritmo módulo 10, 10 dígitos, persona natural)."""
    if len(digits) != 10 or not digits.isdigit():
        return False
    provincia = int(digits[0:2])
    tercer_digito = int(digits[2])
    if not (1 <= provincia <= 24) or tercer_digito >= 6:
        return False
    total = 0
    for coef, d in zip(_CEDULA_COEFFS, digits[:9]):
        val = coef * int(d)
        total += val - 9 if val >= 10 else val
    verificador = (10 - (total % 10)) % 10
    return verificador == int(digits[9])


def _redact_cedula_ruc_by_checksum(s: str) -> str:
    """
    Detecta cédula/RUC sueltos en texto libre (sin label "cedula:"/"RUC:" al lado)
    validando el dígito verificador, en vez de depender de que el agente haya escrito
    la palabra "cedula" pegada al número. Tolera separadores (-, ., espacio) porque
    en la data real aparecen formatos como "091234567-8" o "09 1234567 8".
    Solo redacta si el checksum valida -- evita falsos positivos sobre IDs genéricos
    de 10-13 dígitos que no son cédula/RUC.
    """
    candidate = re.compile(r"\b\d[\d\-\. ]{8,15}\d\b")

    def _repl(m: re.Match) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 10 and _is_valid_cedula_ec(digits):
            return "[CEDULA]"
        if len(digits) == 13 and _is_valid_cedula_ec(digits[:10]) and digits[10:] == "001":
            # RUC de persona natural = cédula + código de establecimiento "001".
            # RUC de sociedad (tercer dígito 6/9) usa otro algoritmo, no cubierto acá.
            return "[RUC]"
        return raw

    return candidate.sub(_repl, s)


def _redact_phone_numbers(s: str) -> str:
    """
    Detecta teléfonos ecuatorianos (móvil 09XXXXXXXX, fijo 0[2-7]XXXXXXX, con o
    sin código de país +593) tolerando separadores (-, ., espacio) en cualquier
    posición. El regex rígido anterior (`0?9\\d{8}`) exigía los dígitos pegados
    y no matcheaba formatos como "09-1234-567-8" que sí aparecen en la data real.
    """
    candidate = re.compile(r"\b\+?\d[\d\-\. ]{6,14}\d\b")

    def _repl(m: re.Match) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if digits.startswith("593"):
            digits = digits[3:]
        if len(digits) == 10 and digits[:2] == "09":
            return "[PHONE]"
        if len(digits) == 9 and digits[0] == "9":
            return "[PHONE]"
        if len(digits) == 9 and digits[0] == "0" and digits[1] in "234567":
            return "[PHONE]"
        if len(digits) == 8 and digits[0] in "234567":
            return "[PHONE]"
        return raw

    return candidate.sub(_repl, s)


def _flatten_text(value: object) -> object:
    # pd.isna can return array-like for some inputs; keep this scalar-safe.
    if value is None:
        return value
    s = str(value)
    if s == "" or s.casefold() == "nan":
        return value
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _anonymize_solution_name_mentions(solution: object, original_name: object) -> object:
    """
    If SOLUTION contains the pattern `NAME: <original_name>`, anonymize the embedded name.
    """
    if solution is None:
        return solution

    sol = str(solution)
    if sol == "" or sol.casefold() == "nan":
        return solution

    if original_name is None:
        return _flatten_text(solution)

    name = str(original_name).strip()
    if name == "" or name.casefold() == "nan":
        return _flatten_text(solution)

    token = _stable_token(name, prefix="NAME")
    pattern = re.compile(rf"(?i)\bNAME:\s*{re.escape(name)}\b")
    sol = pattern.sub(f"NAME: {token}", sol)
    return _flatten_text(sol)


def scrub_free_text(text: object, original_name: object, name_token: str | None) -> object:
    """
    Scrub PII from a free-text field (DESCRIPTION / SOLUTION) without accidentally
    pulling content from other fields.

    - Replaces occurrences of the row's original NAME with its deterministic token (when available)
    - Redacts emails, RUC, cédula, phones, long numeric IDs
    - Flattens newlines to keep one-record-per-line CSV output
    """
    if text is None:
        return text

    s = str(text)
    if s == "" or s.casefold() == "nan":
        return text

    # Replace this row's known user name consistently (Spanish/English labels + plain occurrences).
    if original_name is not None and name_token is not None:
        name = str(original_name).strip()
        if name != "" and name.casefold() != "nan":
            # Labelled pattern first (avoid partial replacements around the label).
            s = re.sub(
                rf"(?i)\b(NOMBRE|NAME)\s*:\s*{re.escape(name)}\b",
                r"\1: " + name_token,
                s,
            )
            # Any remaining occurrences of the exact name.
            s = re.sub(rf"(?i)\b{re.escape(name)}\b", name_token, s)

    # Emails -- dominio multi-nivel (ej. cliente.com.ec), el regex anterior solo
    # cubría un nivel y dejaba el resto del dominio sin redactar.
    s = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}", "[EMAIL]", s)

    # Cédula/RUC con label explícito, tolera separadores y variantes "C.I."/"R.U.C."
    # (el punto en "C.I." rompía el match anterior).
    s = re.sub(r"(?i)(?:c[eé]dula|c\.?i\.?)[:\s]*[\d][\d\-\. ]{8,13}\d", "[CEDULA]", s)
    s = re.sub(r"(?i)r\.?u\.?c\.?[:\s]*[\d][\d\-\. ]{11,16}\d", "[RUC]", s)

    # Cédula/RUC sueltos (sin label), validados por dígito verificador -- ver
    # _redact_cedula_ruc_by_checksum. Corre antes del catch-all de IDs largos.
    s = _redact_cedula_ruc_by_checksum(s)

    # RUC exacto de 13 dígitos que no pasó el checksum (ej. RUC de sociedad,
    # algoritmo distinto no implementado acá) -- se redacta igual por precaución.
    s = re.sub(r"\b\d{13}\b", "[RUC]", s)

    # Teléfonos (móvil/fijo, con o sin separadores/código de país) -- ver
    # _redact_phone_numbers.
    s = _redact_phone_numbers(s)

    # Long numeric IDs (catch-all final, ya sin los que quedaron marcados arriba)
    s = re.sub(r"\b\d{8,}\b", "[ID]", s)

    return _flatten_text(s)



def load_db_files(dir_path: Path = DIR_PATH) -> Dict[str, pd.DataFrame]:
    """
    Carga todos los datasets disponibles en `dir_path` y los retorna como un dict.

    Soporta:
    - `.txt` delimitados por `|`
    - `.xlsx`
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    datasets: Dict[str, pd.DataFrame] = {}

    for p in sorted(dir_path.iterdir()):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()

        if suffix == ".txt":
            datasets[p.stem] = pd.read_csv(
                p,
                sep="|",
                low_memory=False,
                encoding="utf-8",
                encoding_errors="replace",
            )
        elif suffix == ".xlsx":
            datasets[p.stem] = pd.read_excel(p, engine="openpyxl")

    return datasets


def load_claims(dir_path: Path = DIR_PATH) -> pd.DataFrame:
    """
    Carga únicamente el dataset de claims desde `dir_path`.

    Prioridad:
    1) claims.csv (si existe)
    2) Example_full_claims.txt (pipe-delimited)
    3) claims.xlsx (si existe)
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    claims_csv = dir_path / "claims.csv"
    if claims_csv.exists():
        return pd.read_csv(
            claims_csv,
            sep="|",
            engine="python",
            encoding="utf-8",
            encoding_errors="replace",
        )

    claims_txt = dir_path / "Example_full_claims.txt"
    if claims_txt.exists():
        return pd.read_csv(
            claims_txt,
            sep="|",
            engine="python",
            encoding="utf-8",
            encoding_errors="replace",
        )

    claims_xlsx = dir_path / "claims.xlsx"
    if claims_xlsx.exists():
        return pd.read_excel(claims_xlsx, engine="openpyxl")

    raise FileNotFoundError(
        f"No claims file found in {dir_path} (expected claims.csv, claims.xlsx, or Example_full_claims.txt)"
    )


def load_location(dir_path: Path = DIR_PATH) -> pd.DataFrame:
    """
    Loads the location dataset from `dir_path`.
    Expects location.csv or location.txt (pipe-delimited).
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for filename in ("location.csv", "location.txt"):
        p = dir_path / filename
        if p.exists():
            return pd.read_csv(
                p,
                sep="|",
                low_memory=False,
                encoding="utf-8",
                encoding_errors="replace",
            )

    raise FileNotFoundError(
        f"No location file found in {dir_path} (expected location.csv or location.txt)"
    )


def process_claims(
    df: pd.DataFrame,
    output_csv_path: Path,
) -> int:
    """
    Anonymizes PII in a claims DataFrame row-by-row and writes to CSV.
    Returns number of data rows written.
    """

    cols_upper = {str(c).upper(): c for c in df.columns}
    structured_present = [c for name, c in cols_upper.items() if name in set(STRUCTURED_PII_COLS)]

    desc_col = cols_upper.get("DESCRIPTION")
    sol_col = cols_upper.get("SOLUTION")
    name_col = cols_upper.get("NAME")
    nom_usuario_col = cols_upper.get("NOM_USUARIO")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    # Use UTF-8 with BOM for better compatibility with Excel.
    with output_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[str(c) for c in df.columns], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for _, row in df.iterrows():
            out: dict[str, object] = {str(c): row[c] for c in df.columns}

            # Use NAME as the single key for user-name anonymization, and replicate it to NOM_USUARIO.
            original_name = out[str(name_col)] if name_col is not None else None
            name_token: str | None = None
            if original_name is not None:
                s = str(original_name).strip()
                if s != "" and s.casefold() != "nan":
                    name_token = _stable_token(s, prefix="NAME")

            if name_token is not None:
                if name_col is not None:
                    out[str(name_col)] = name_token
                if nom_usuario_col is not None:
                    out[str(nom_usuario_col)] = name_token

            # Structured PII: deterministic anonymization
            for c in structured_present:
                # NAME and NOM_USUARIO are handled above using the NAME token.
                if name_col is not None and c == name_col:
                    continue
                if nom_usuario_col is not None and c == nom_usuario_col:
                    continue
                v = out[str(c)]
                if v is not None and str(v).strip() != "" and str(v).casefold() != "nan":
                    out[str(c)] = _stable_token(v, prefix=str(c).upper())

            # Free-text scrubbing + formatting
            if desc_col is not None:
                out[str(desc_col)] = scrub_free_text(out[str(desc_col)], original_name, name_token)
            if sol_col is not None:
                # Always use the original NAME value as key for replacement.
                out[str(sol_col)] = scrub_free_text(out[str(sol_col)], original_name, name_token)

            writer.writerow(out)
            written += 1

    return written


def main() -> None:
    xlsx_path = DIR_PATH / "claims.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Expected file not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    out_path = (BASE_DIR / "output" / "claims_clean.csv").resolve()
    n = process_claims(df, out_path)

    print(f"Processed rows: {n}")
    print(f"Wrote cleaned CSV to: {out_path}")


if __name__ == "__main__":
    main()

