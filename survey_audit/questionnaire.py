"""Extract the U.S. ISSP questionnaire using the original block parser."""

import re
import pymupdf

QUESTION_SOURCES = {'v33': ('PAYTAXES', 'PAYTAXES', 'Never to try to evade taxes'), 'v37': ('CONSCRP', 'PAYTAXES', 'To be willing to serve'), 'v39': ('DEM10PST_', 'DEMTODAY_', 'And what about 10 years ago?'), 'v43': ('POLOPTS', 'SOLOK', 'That people be given'), 'v28': ('IMMJOBS1', 'IMMCRIME1', 'Immigrants take jobs'), 'v32': ('IMMASSIM', '', ''), 'v27': ('IMMAMECO', 'IMMCRIME1', 'Immigrants are generally'), 'v20': ('IMPORTS', 'IMPORTS', 'America should limit'), 'v25': ('INTLINCS', 'IMPORTS', 'Large international companies'), 'v24': ('AMTV', 'IMPORTS', 'American television'), 'v59': ('CANTRUST', '', ''), 'v58': ('BEFAIR', '', ''), 'v6': ('AMANCSTR', 'AMBORNIN', 'To have American ancestry'), 'v4': ('AMGOVT', 'AMBORNIN', 'To respect American'), 'v8': ('AMSHAMED', 'AMCITIZN', 'There are some things'), 'v16': ('PROUDECO', 'PROUDDEM', 'America’s economic'), 'v18': ('PROUDART', 'PROUDDEM', 'Its achievements'), 'v17': ('PROUDSPT', 'PROUDDEM', 'Its achievements'), 'v47': ('POLEFF11', 'POLEFF11', 'People like me'), 'v48': ('POLEFF18', 'POLEFF11', 'I don’t think'), 'v57': ('BIASNEWS', 'BIASNEWS', 'Media coverage'), 'v56': ('POLNEWS', '', ''), 'v46': ('LEFTRGHT', '', ''), 'v55': ('EXPERTS', 'FULLDEM', 'It is better'), 'v52': ('POLTALK', 'FULLDEM', 'Politicians talk'), 'v54': ('BIZMONEY', 'FULLDEM', 'Business leaders'), 'v50': ('CORRUPTN', '', '')}

BATTERY_ITEM_STARTS = {'PAYTAXES': 'Never to try to evade taxes', 'DEMTODAY_': 'How well does democracy work', 'SOLOK': 'That all citizens', 'IMMCRIME1': 'Immigrants increase crime rates.', 'IMPORTS': 'America should limit', 'AMBORNIN': 'To have been born', 'AMCITIZN': 'I would rather', 'PROUDDEM': 'The way democracy works', 'POLEFF11': 'People like me', 'BIASNEWS': 'Media coverage', 'FULLDEM': 'The people, and not politicians,'}

def normalize_pdf_text(text: str) -> str:
    """Collapse PDF layout whitespace without changing words or punctuation."""
    return ' '.join(text.split())

def read_questionnaire(questionnaire_path) -> dict[str, dict]:
    """Extract GSS question blocks and category labels from the supplied PDF."""
    pages = []
    with pymupdf.open(questionnaire_path) as document:
        for page in document:
            lines = page.get_text().splitlines()
            pages.append('\n'.join((line for line in lines if not line.startswith('MDDtoDOC - '))))
    text = '\n'.join(pages)
    page_ends = []
    position = 0
    for page in pages:
        position += len(page) + 1
        page_ends.append(position)
    headings = list(re.finditer('(?m)^([A-Z][A-Z0-9_]*): Categorical \\(Single\\)', text))
    questions = {}
    for (position, heading) in enumerate(headings):
        end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        block = text[heading.end():end]
        if 'Categories:' not in block:
            continue
        (question, categories) = block.split('Categories:', 1)
        options = []
        for category in re.finditer('\\{([^{}]+)\\}([^{}]*)', categories):
            (key, label) = category.groups()
            if key.lower() in {'refused', 'noanswer'}:
                break
            options.append((key, normalize_pdf_text(label)))
        questions[heading.group(1)] = {'question': normalize_pdf_text(question), 'options': options, 'pdf_page': next((i + 1 for (i, end) in enumerate(page_ends) if heading.start() < end))}
    return questions
