#!/usr/bin/env python3
"""
Zero-dependency LaTeX to MathML converter for Python.
Generates W3C-compliant MathML natively rendered by modern browsers (Chrome 109+, Safari, Firefox)
without ANY client-side JavaScript.
"""

import html
import re

GREEK_LETTERS = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ϵ',
    'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'vartheta': 'ϑ',
    'iota': 'ι', 'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ',
    'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'ϕ',
    'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ',
    'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω'
}

OPERATORS = {
    'sum': '∑',
    'prod': '∏',
    'coprod': '∐',
    'int': '∫',
    'iint': '∬',
    'iiint': '∭',
    'oint': '∮',
    'nabla': '∇',
    'partial': '∂',
    'infty': '∞',
    'times': '×',
    'cdot': '⋅',
    'circ': '∘',
    'bullet': '∙',
    'pm': '±',
    'mp': '∓',
    'le': '≤',
    'leq': '≤',
    'ge': '≥',
    'geq': '≥',
    'ne': '≠',
    'neq': '≠',
    'approx': '≈',
    'sim': '∼',
    'equiv': '≡',
    'propto': '∝',
    'in': '∈',
    'notin': '∉',
    'subset': '⊂',
    'subseteq': '⊆',
    'supset': '⊃',
    'supseteq': '⊇',
    'cap': '∩',
    'cup': '∪',
    'to': '→',
    'rightarrow': '→',
    'leftarrow': '←',
    'Rightarrow': '⇒',
    'Leftarrow': '⇐',
    'iff': '⟺',
    'leftrightarrow': '↔',
    'mapsto': '↦',
    'dots': '…',
    'ldots': '…',
    'cdots': '⋯',
    'vdots': '⋮',
    'ddots': '⋱',
    'forall': '∀',
    'exists': '∃',
    'nexists': '∄',
    'odot': '⊙',
    'oplus': '⊕',
    'otimes': '⊗',
    'top': '⊤',
    'bot': '⊥',
    'mid': '∣',
    'langle': '⟨',
    'rangle': '⟩',
    'Vert': '‖',
    'vert': '|',
    'prime': '′',
    'colon': ':'
}

NAMED_FUNCTIONS = {
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'exp', 'log', 'ln', 'lg',
    'det', 'dim', 'ker', 'deg', 'gcd', 'hom', 'Pr',
    'softmax', 'Softmax', 'Attention', 'Swish', 'Concat',
    'LayerNorm', 'RMSNorm', 'RMS', 'FFN', 'Var', 'Cov',
    'argmax', 'argmin', 'max', 'min', 'sup', 'inf', 'lim'
}

SPECIAL_CHARS = {
    '{': '{', '}': '}', '%': '%', '$': '$', '&': '&', '#': '#', '_': '_'
}

LARGE_OPERATORS = {'sum', 'prod', 'coprod', 'int', 'iint', 'iiint', 'oint', 'lim', 'max', 'min', 'argmax', 'argmin'}


class Token:
    def __init__(self, typ, val):
        self.type = typ
        self.val = val

    def __repr__(self):
        return f"Token({self.type}, {repr(self.val)})"


def tokenize(latex):
    """Tokenize a LaTeX math string."""
    tokens = []
    i = 0
    n = len(latex)

    while i < n:
        c = latex[i]

        # Skip whitespace
        if c.isspace():
            i += 1
            continue

        # LaTeX Commands
        if c == '\\':
            i += 1
            if i >= n:
                break
            # Escaped special symbol like \{ or \} or \%
            if latex[i] in SPECIAL_CHARS or latex[i] in ['\\', ',', ';', '!', ' ', '|']:
                cmd = latex[i]
                i += 1
                tokens.append(Token('COMMAND', cmd))
                continue
            
            # Named command letters
            start = i
            while i < n and latex[i].isalpha():
                i += 1
            cmd = latex[start:i]
            tokens.append(Token('COMMAND', cmd))
            continue

        # Braces
        if c in '{}[()]':
            tokens.append(Token('BRACE', c))
            i += 1
            continue

        # Sub / Sup
        if c in '^_':
            tokens.append(Token('SCRIPT', c))
            i += 1
            continue

        # Numbers (including decimals)
        if c.isdigit():
            start = i
            while i < n and (latex[i].isdigit() or latex[i] == '.'):
                i += 1
            tokens.append(Token('NUMBER', latex[start:i]))
            continue

        # Identifier letters
        if c.isalpha():
            tokens.append(Token('IDENTIFIER', c))
            i += 1
            continue

        # Punctuation & Operators
        if c in '+-*/=<>!~|.,;:?`\'"':
            # Check for multi-char operators like <= or >= or ->
            tokens.append(Token('OPERATOR', c))
            i += 1
            continue

        # Fallback
        tokens.append(Token('CHAR', c))
        i += 1

    return tokens


class MathParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        tok = self.peek()
        if tok:
            self.pos += 1
        return tok

    def expect(self, typ, val=None):
        tok = self.peek()
        if not tok or tok.type != typ or (val is not None and tok.val != val):
            return None
        return self.consume()

    def parse(self):
        nodes = []
        while self.peek():
            node = self.parse_primary()
            if node:
                nodes.append(node)
            else:
                self.consume()
        return nodes

    def parse_group_or_item(self):
        """Parse a single item or a { ... } group."""
        tok = self.peek()
        if not tok:
            return None
        if tok.type == 'BRACE' and tok.val == '{':
            self.consume()
            nodes = []
            while self.peek() and not (self.peek().type == 'BRACE' and self.peek().val == '}'):
                node = self.parse_primary()
                if node:
                    nodes.append(node)
                else:
                    self.consume()
            self.expect('BRACE', '}')
            if len(nodes) == 1:
                return nodes[0]
            return ('mrow', nodes)
        else:
            return self.parse_single_atom()

    def parse_single_atom(self):
        tok = self.peek()
        if not tok:
            return None
        
        # Check for commands like \mathbf, \text, \frac, etc.
        if tok.type == 'COMMAND':
            cmd = tok.val
            self.consume()
            
            # Spaces
            if cmd in [',', ';', ':', ' ', 'quad', 'qquad']:
                width = '1em' if cmd == 'quad' else ('2em' if cmd == 'qquad' else '0.25em')
                return ('mspace', width)
            
            # Greek letters
            if cmd in GREEK_LETTERS:
                return ('mi', GREEK_LETTERS[cmd])
            
            # Named functions (e.g. \sin, \log)
            if cmd in NAMED_FUNCTIONS:
                return ('mi_func', cmd)
            
            # Math operators
            if cmd in OPERATORS:
                is_large = cmd in LARGE_OPERATORS
                return ('mo_op', OPERATORS[cmd], is_large)
            
            # Styling commands: \mathbf, \mathbb, \mathcal, \boldsymbol, \mathit, \mathrm
            if cmd in ['mathbf', 'boldsymbol', 'mathbb', 'mathcal', 'mathit', 'mathrm', 'bm']:
                inner = self.parse_group_or_item()
                variant = 'bold'
                if cmd == 'mathbb':
                    variant = 'double-struck'
                elif cmd == 'mathcal':
                    variant = 'script'
                elif cmd == 'mathit':
                    variant = 'italic'
                elif cmd == 'mathrm':
                    variant = 'normal'
                return ('style', variant, inner)
            
            # Text command: \text, \operatorname
            if cmd in ['text', 'operatorname', 'rm']:
                tok_next = self.peek()
                if tok_next and tok_next.type == 'BRACE' and tok_next.val == '{':
                    self.consume()
                    # Collect all tokens until closing '}' as raw text
                    txt_parts = []
                    brace_depth = 1
                    while self.peek() and brace_depth > 0:
                        t = self.consume()
                        if t.type == 'BRACE' and t.val == '{':
                            brace_depth += 1
                            txt_parts.append('{')
                        elif t.type == 'BRACE' and t.val == '}':
                            brace_depth -= 1
                            if brace_depth > 0:
                                txt_parts.append('}')
                        elif t.type == 'COMMAND':
                            txt_parts.append(t.val)
                        else:
                            txt_parts.append(t.val)
                    return ('mtext', "".join(txt_parts))
                else:
                    item = self.parse_group_or_item()
                    return ('mtext', str(item))

            # Fraction: \frac{num}{den}
            if cmd == 'frac':
                num = self.parse_group_or_item()
                den = self.parse_group_or_item()
                return ('mfrac', num, den)

            # Square root: \sqrt{arg} or \sqrt[n]{arg}
            if cmd == 'sqrt':
                if self.peek() and self.peek().type == 'BRACE' and self.peek().val == '[':
                    self.consume()
                    root_deg = []
                    while self.peek() and not (self.peek().type == 'BRACE' and self.peek().val == ']'):
                        root_deg.append(self.parse_primary())
                    self.expect('BRACE', ']')
                    radicand = self.parse_group_or_item()
                    return ('mroot', radicand, ('mrow', root_deg))
                else:
                    radicand = self.parse_group_or_item()
                    return ('msqrt', radicand)

            # \left and \right delimiters
            if cmd == 'left':
                delim = self.consume()
                d_val = delim.val if delim else '('
                if d_val == '{':
                    d_val = '{'
                elif d_val == '.':
                    d_val = ''
                body = []
                while self.peek():
                    if self.peek().type == 'COMMAND' and self.peek().val == 'right':
                        self.consume()
                        r_delim = self.consume()
                        rd_val = r_delim.val if r_delim else ')'
                        if rd_val == '}':
                            rd_val = '}'
                        elif rd_val == '.':
                            rd_val = ''
                        return ('delim', d_val, rd_val, ('mrow', body))
                    body.append(self.parse_primary())
                return ('delim', d_val, ')', ('mrow', body))

            # Escaped symbols like \{, \}, \|
            if cmd in ['{', '}']:
                return ('mo', cmd)
            if cmd in ['|', 'Vert']:
                return ('mo', '‖')

            # Unknown command - render as text
            return ('mi', cmd)

        # Number
        if tok.type == 'NUMBER':
            self.consume()
            return ('mn', tok.val)

        # Identifier
        if tok.type == 'IDENTIFIER':
            self.consume()
            return ('mi', tok.val)

        # Operator
        if tok.type == 'OPERATOR':
            self.consume()
            val = tok.val
            if val == '<':
                val = '<'
            elif val == '>':
                val = '>'
            return ('mo', val)

        # Braces / Parentheses
        if tok.type == 'BRACE':
            if tok.val == '{':
                return self.parse_group_or_item()
            elif tok.val == '}':
                self.consume()
                return None
            else:
                self.consume()
                return ('mo', tok.val)

        self.consume()
        return ('mo', tok.val)

    def parse_primary(self):
        """Parse an atom followed by optional superscripts / subscripts."""
        base = self.parse_single_atom()
        if not base:
            return None

        sub = None
        sup = None

        while self.peek() and self.peek().type == 'SCRIPT':
            tok = self.consume()
            if tok.val == '_':
                sub = self.parse_group_or_item()
            elif tok.val == '^':
                sup = self.parse_group_or_item()

        if sub is not None and sup is not None:
            # Check if base is a large operator (sum, prod, lim, etc.)
            if base[0] == 'mo_op' and base[2]:
                return ('munderover', base, sub, sup)
            return ('msubsup', base, sub, sup)
        elif sub is not None:
            if base[0] == 'mo_op' and base[2]:
                return ('munder', base, sub)
            return ('msub', base, sub)
        elif sup is not None:
            if base[0] == 'mo_op' and base[2]:
                return ('mover', base, sup)
            return ('msup', base, sup)

        return base


def to_mathml_xml(node):
    """Render an AST node into MathML XML."""
    if node is None:
        return ""
    
    ntype = node[0]

    if ntype == 'mi':
        val = html.escape(node[1])
        return f"<mi>{val}</mi>"
    
    if ntype == 'mi_func':
        val = html.escape(node[1])
        return f"<mi>{val}</mi>"

    if ntype == 'mn':
        val = html.escape(node[1])
        return f"<mn>{val}</mn>"

    if ntype == 'mo':
        val = html.escape(node[1])
        return f"<mo>{val}</mo>"

    if ntype == 'mo_op':
        sym = html.escape(node[1])
        large = node[2]
        if large:
            return f'<mo movablelimits="true">{sym}</mo>'
        return f"<mo>{sym}</mo>"

    if ntype == 'mtext':
        val = html.escape(node[1])
        return f"<mtext>{val}</mtext>"

    if ntype == 'mspace':
        width = node[1]
        return f'<mspace width="{width}"/>'

    if ntype == 'style':
        variant = node[1]
        inner = node[2]
        # In MathML, characters can have mathvariant="bold", "double-struck", "script"
        if inner and inner[0] == 'mi':
            # e.g. <mi mathvariant="bold">x</mi>
            val = html.escape(inner[1])
            return f'<mi mathvariant="{variant}">{val}</mi>'
        else:
            return f'<mstyle mathvariant="{variant}">{to_mathml_xml(inner)}</mstyle>'

    if ntype == 'mrow':
        inner = "".join(to_mathml_xml(c) for c in node[1])
        return f"<mrow>{inner}</mrow>"

    if ntype == 'mfrac':
        num = to_mathml_xml(node[1])
        den = to_mathml_xml(node[2])
        return f"<mfrac><mrow>{num}</mrow><mrow>{den}</mrow></mfrac>"

    if ntype == 'msqrt':
        rad = to_mathml_xml(node[1])
        return f"<msqrt><mrow>{rad}</mrow></msqrt>"

    if ntype == 'mroot':
        rad = to_mathml_xml(node[1])
        deg = to_mathml_xml(node[2])
        return f"<mroot><mrow>{rad}</mrow><mrow>{deg}</mrow></mroot>"

    if ntype == 'msub':
        base = to_mathml_xml(node[1])
        sub = to_mathml_xml(node[2])
        return f"<msub>{base}{sub}</msub>"

    if ntype == 'msup':
        base = to_mathml_xml(node[1])
        sup = to_mathml_xml(node[2])
        return f"<msup>{base}{sup}</msup>"

    if ntype == 'msubsup':
        base = to_mathml_xml(node[1])
        sub = to_mathml_xml(node[2])
        sup = to_mathml_xml(node[3])
        return f"<msubsup>{base}{sub}{sup}</msubsup>"

    if ntype == 'munder':
        base = to_mathml_xml(node[1])
        under = to_mathml_xml(node[2])
        return f"<munder>{base}{under}</munder>"

    if ntype == 'mover':
        base = to_mathml_xml(node[1])
        over = to_mathml_xml(node[2])
        return f"<mover>{base}{over}</mover>"

    if ntype == 'munderover':
        base = to_mathml_xml(node[1])
        under = to_mathml_xml(node[2])
        over = to_mathml_xml(node[3])
        return f"<munderover>{base}{under}{over}</munderover>"

    if ntype == 'delim':
        ldelim = html.escape(node[1])
        rdelim = html.escape(node[2])
        inner = to_mathml_xml(node[3])
        l_tag = f"<mo>{ldelim}</mo>" if ldelim else ""
        r_tag = f"<mo>{rdelim}</mo>" if rdelim else ""
        return f"<mrow>{l_tag}{inner}{r_tag}</mrow>"

    return ""


def latex_to_mathml(latex_str, display=False):
    """
    Translates a LaTeX math string into a valid MathML `<math>` block.
    Works natively across all modern browsers with 0 JavaScript.
    """
    latex_str = latex_str.strip()
    # Clean up double backslashes or artifacts
    tokens = tokenize(latex_str)
    parser = MathParser(tokens)
    nodes = parser.parse()
    inner_xml = "".join(to_mathml_xml(n) for n in nodes)
    
    disp_attr = ' display="block"' if display else ' display="inline"'
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML"{disp_attr}>{inner_xml}</math>'


if __name__ == "__main__":
    sample = r"\sum_{i=1}^N \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|} = \prod_{t=1}^T P(w_t \mid w_{<t})"
    print(latex_to_mathml(sample, display=True))
