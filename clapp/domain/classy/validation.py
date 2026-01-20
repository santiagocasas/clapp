import ast
import difflib
import os
import re

_CLASSY_VALID_PARAMS = None
_CLASSY_VERBOSE_MODULES = {
    "input",
    "background",
    "thermodynamics",
    "perturbations",
    "transfer",
    "primordial",
    "harmonic",
    "fourier",
    "lensing",
    "distortions",
    "output",
}

_CLASSY_TOKEN_EXCLUSIONS = {
    "and",
    "as",
    "by",
    "default",
    "defaults",
    "for",
    "from",
    "in",
    "of",
    "or",
    "with",
}


def _add_token(valid: set, token: str):
    token = token.strip().strip("'\"`).,;:()[]{}")
    if not token:
        return
    if token.lower() in _CLASSY_TOKEN_EXCLUSIONS:
        return
    if re.fullmatch(r"[A-Za-z0-9_./-]+", token):
        valid.add(token)


def _extract_tokens_from_line(line: str, valid: set):
    for token in re.findall(r"``([^`]+)``", line):
        _add_token(valid, token)

    for token in re.findall(r"\b([A-Za-z0-9_./-]+)\s*=", line):
        _add_token(valid, token)

    bullet_match = re.match(r"\s*[*-]\s*(.+)", line)
    if bullet_match:
        bullet = bullet_match.group(1)
        for part in re.split(r"\s+or\s+", bullet):
            token_match = re.match(r"[A-Za-z0-9_./-]+", part.strip())
            if token_match:
                _add_token(valid, token_match.group(0))


def get_classy_valid_params():
    global _CLASSY_VALID_PARAMS
    if _CLASSY_VALID_PARAMS is not None:
        return _CLASSY_VALID_PARAMS
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "class-data",
    )
    valid = set()
    if not os.path.isdir(data_dir):
        _CLASSY_VALID_PARAMS = valid
        return valid
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".ini"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    lines = file.read().splitlines()
            except OSError:
                continue
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", ";", "[")):
                    continue
                if "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key and re.fullmatch(r"[A-Za-z0-9_./-]+", key):
                        valid.add(key)
            continue
        if not filename.endswith((".txt", ".md", ".rst")):
            continue
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.read().splitlines()
        except OSError:
            continue
        for line in lines:
            _extract_tokens_from_line(line, valid)
    _CLASSY_VALID_PARAMS = valid
    return valid


def _extract_literal_dict_keys(node: ast.Dict):
    keys = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def extract_class_params_from_code(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    dict_assigns = {}
    found = set()

    class ParamVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Dict):
                keys = _extract_literal_dict_keys(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        dict_assigns[target.id] = keys
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if isinstance(node.value, ast.Dict) and isinstance(node.target, ast.Name):
                dict_assigns[node.target.id] = _extract_literal_dict_keys(node.value)
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set":
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Dict):
                        found.update(_extract_literal_dict_keys(arg))
                    elif isinstance(arg, ast.Name) and arg.id in dict_assigns:
                        found.update(dict_assigns[arg.id])
                for kw in node.keywords:
                    if kw.arg:
                        found.add(kw.arg)
            self.generic_visit(node)

    ParamVisitor().visit(tree)
    return found


def validate_class_params_in_code(code: str):
    used_params = extract_class_params_from_code(code)
    if not used_params:
        return [], {}
    valid_params = get_classy_valid_params()
    invalid = []
    suggestions = {}
    for param in sorted(used_params):
        if param in valid_params:
            continue
        if param.endswith("_verbose"):
            prefix = param[:-8]
            if prefix in _CLASSY_VERBOSE_MODULES:
                continue
        invalid.append(param)
        suggestions[param] = difflib.get_close_matches(
            param, valid_params, n=3, cutoff=0.6
        )
    return invalid, suggestions


def auto_correct_class_params(code: str, suggestions: dict):
    corrected = code
    replacements = []
    for param, candidates in suggestions.items():
        if not candidates:
            continue
        suggestion = candidates[0]
        pattern = r"(['\"])%s\\1" % re.escape(param)
        corrected, count = re.subn(pattern, r"\\1%s\\1" % suggestion, corrected)
        if count:
            replacements.append((param, suggestion))
    return corrected, replacements
