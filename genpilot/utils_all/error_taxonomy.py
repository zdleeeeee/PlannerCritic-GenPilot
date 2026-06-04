import json
import re

OBJECT_ERROR = "object_error"
ATTRIBUTE_ERROR = "attribute_error"
SPATIAL_RELATION_ERROR = "spatial_relation_error"
COUNTING_ERROR = "counting_error"
ACTION_ERROR = "action_error"
STYLE_ERROR = "style_error"
TEXT_RENDERING_ERROR = "text_rendering_error"
OTHER_ERROR = "other_error"

CANONICAL_CATEGORIES = {
    OBJECT_ERROR,
    ATTRIBUTE_ERROR,
    SPATIAL_RELATION_ERROR,
    COUNTING_ERROR,
    ACTION_ERROR,
    STYLE_ERROR,
    TEXT_RENDERING_ERROR,
    OTHER_ERROR,
}

CATEGORY_ALIASES = {
    "object": OBJECT_ERROR,
    "object error": OBJECT_ERROR,
    "object_error": OBJECT_ERROR,
    "missing object": OBJECT_ERROR,
    "wrong object": OBJECT_ERROR,
    "attribute": ATTRIBUTE_ERROR,
    "attribute error": ATTRIBUTE_ERROR,
    "attribute_error": ATTRIBUTE_ERROR,
    "color error": ATTRIBUTE_ERROR,
    "material error": ATTRIBUTE_ERROR,
    "spatial": SPATIAL_RELATION_ERROR,
    "spatial error": SPATIAL_RELATION_ERROR,
    "spatial relation": SPATIAL_RELATION_ERROR,
    "spatial relation error": SPATIAL_RELATION_ERROR,
    "spatial_relation_error": SPATIAL_RELATION_ERROR,
    "relationship error": SPATIAL_RELATION_ERROR,
    "count": COUNTING_ERROR,
    "counting": COUNTING_ERROR,
    "counting error": COUNTING_ERROR,
    "counting_error": COUNTING_ERROR,
    "number error": COUNTING_ERROR,
    "quantity error": COUNTING_ERROR,
    "action": ACTION_ERROR,
    "action error": ACTION_ERROR,
    "action_error": ACTION_ERROR,
    "pose error": ACTION_ERROR,
    "activity error": ACTION_ERROR,
    "style": STYLE_ERROR,
    "style error": STYLE_ERROR,
    "sytle error": STYLE_ERROR,
    "sytle": STYLE_ERROR,
    "style_error": STYLE_ERROR,
    "aesthetic error": STYLE_ERROR,
    "text": TEXT_RENDERING_ERROR,
    "text error": TEXT_RENDERING_ERROR,
    "text rendering": TEXT_RENDERING_ERROR,
    "text rendering error": TEXT_RENDERING_ERROR,
    "text_rendering_error": TEXT_RENDERING_ERROR,
    "typography error": TEXT_RENDERING_ERROR,
    "other": OTHER_ERROR,
    "other error": OTHER_ERROR,
    "other_error": OTHER_ERROR,
}

CATEGORY_PATTERNS = [
    (
        COUNTING_ERROR,
        [
            r"\b(exactly|count|number of|quantity|how many|too many|too few|fewer|more than|less than)\b",
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            r"\b\d+\b",
        ],
    ),
    (
        TEXT_RENDERING_ERROR,
        [
            r"\b(text|letter|word|sign|label|logo|caption|typography|font|written|writing|readable|legible|spelling|misspelled)\b",
            r"[\"“”'‘’][^\"“”'‘’]{1,40}[\"“”'‘’]",
        ],
    ),
    (
        SPATIAL_RELATION_ERROR,
        [
            r"\b(left|right|above|below|under|over|front|behind|beside|next to|between|inside|outside|around|near|far|overlap|spatial|position|located|placement|relationship)\b",
        ],
    ),
    (
        ACTION_ERROR,
        [
            r"\b(action|activity|doing|pose|posing|gesture|holding|wearing|carrying|riding|running|walking|jumping|sitting|standing|looking|eating|drinking|playing|flying)\b",
        ],
    ),
    (
        STYLE_ERROR,
        [
            r"\b(style|aesthetic|art style|medium|painting|photo|photograph|illustration|cartoon|anime|cinematic|lighting|camera|lens|close-up|wide shot|composition|mood|tone|color palette)\b",
        ],
    ),
    (
        ATTRIBUTE_ERROR,
        [
            r"\b(color|colou?r|red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|material|wooden|metal|glass|shape|size|texture|pattern|striped|spotted|round|square|small|large|tall|short|crinkled|plump|dense|soft|warm|golden)\b",
        ],
    ),
    (
        OBJECT_ERROR,
        [
            r"\b(missing|absent|not present|does not show|no visible|wrong object|incorrect object|extra object|additional object|replaced|instead of|not include|omits|lacks)\b",
            r"\b(object|subject|person|people|animal|building|tree|car|cabbage|flower|mountain|lake|field|forest)\b",
        ],
    ),
]

STRATEGIES = {
    OBJECT_ERROR: "Object error: explicitly name the missing or wrong object, keep the same noun throughout, and remove ambiguous synonyms that could let the model substitute another object.",
    ATTRIBUTE_ERROR: "Attribute error: bind each color, material, shape, texture, size, or lighting attribute immediately next to the exact object it modifies.",
    SPATIAL_RELATION_ERROR: "Spatial relation error: name both related objects and use direct layout words such as left, right, front, behind, above, below, inside, or between.",
    COUNTING_ERROR: "Counting error: state the exact cardinality with a numeral and word when useful, avoid vague plurals, and repeat the count only near the target object.",
    ACTION_ERROR: "Action error: use an active verb phrase with a clear subject and object, and avoid passive or narrative wording that hides who is doing what.",
    STYLE_ERROR: "Style error: place medium, lighting, camera, composition, or aesthetic constraints in a clear global style clause without changing scene content.",
    TEXT_RENDERING_ERROR: "Text rendering error: quote the exact visible text, require it to be legible, and explicitly forbid extra or misspelled text.",
    OTHER_ERROR: "Other error: make the smallest unambiguous edit that directly addresses the stated mismatch while preserving the original sentence.",
}

AGGREGATION_PRIORITY = [
    OBJECT_ERROR,
    COUNTING_ERROR,
    TEXT_RENDERING_ERROR,
    SPATIAL_RELATION_ERROR,
    ACTION_ERROR,
    ATTRIBUTE_ERROR,
    STYLE_ERROR,
    OTHER_ERROR,
]

_EMPTY_ERROR_TEXTS = {"", "none", "no error", "no errors", "no_error", "null", "n/a"}


def normalize_categories(value):
    if value is None:
        return [OTHER_ERROR]
    items = _coerce_to_items(value)
    categories = []
    for item in items:
        key = _normalize_key(str(item))
        category = CATEGORY_ALIASES.get(key)
        if category is None and key in CANONICAL_CATEGORIES:
            category = key
        if category and category not in categories:
            categories.append(category)
    return categories or [OTHER_ERROR]


def classify_error(error_text, sentence=""):
    text = f"{error_text or ''}\n{sentence or ''}".lower()
    if not text.strip() or text.strip() in {"none", "no error", "no errors"}:
        return [OTHER_ERROR]

    categories = []
    for category, patterns in CATEGORY_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            categories.append(category)

    if OBJECT_ERROR in categories and len(categories) > 1:
        categories.remove(OBJECT_ERROR)
        categories.insert(0, OBJECT_ERROR)

    return categories or [OTHER_ERROR]


def build_strategy_block(categories):
    normalized = normalize_categories(categories)
    lines = [STRATEGIES[category] for category in normalized]
    if OTHER_ERROR not in normalized:
        lines.append("Keep the edit minimal: change only the phrase needed to fix these error categories.")
    return "\n".join(f"- {line}" for line in lines)


def aggregate_errors(errors, sentence="", max_errors_per_fragment=5):
    error_texts = _flatten_error_texts(errors)
    if len(error_texts) <= max_errors_per_fragment:
        return error_texts

    grouped = {category: [] for category in AGGREGATION_PRIORITY}
    for error_text in error_texts:
        categories = classify_error(error_text, sentence)
        category = _highest_priority_category(categories)
        grouped[category].append(error_text)

    aggregated = []
    for category in AGGREGATION_PRIORITY:
        if grouped[category]:
            aggregated.append(max(grouped[category], key=lambda item: (len(item), item)))

    return aggregated[:max_errors_per_fragment]


def count_error_items(errors):
    return len(_flatten_error_texts(errors))


def _highest_priority_category(categories):
    normalized = normalize_categories(categories)
    specific_categories = [category for category in normalized if category != OBJECT_ERROR]
    if specific_categories:
        normalized = specific_categories
    return min(normalized, key=lambda category: AGGREGATION_PRIORITY.index(category) if category in AGGREGATION_PRIORITY else len(AGGREGATION_PRIORITY))


def _flatten_error_texts(value):
    items = []
    for item in _walk_error_items(value):
        text = _format_error_item(item)
        if _is_meaningful_error(text):
            items.append(text)
    deduped = []
    seen = set()
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _walk_error_items(value):
    if value is None:
        return
    if isinstance(value, dict):
        if any(key in value for key in ("error", "errors", "description", "explanation", "text", "message", "category", "type")):
            yield value
            return
        for item in value.values():
            yield from _walk_error_items(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk_error_items(item)
        return
    if isinstance(value, str):
        parsed = _try_parse_json(value)
        if parsed is not value:
            yield from _walk_error_items(parsed)
            return
        for part in re.split(r"\n+", value):
            yield part
        return
    yield value


def _format_error_item(item):
    if isinstance(item, dict):
        category = item.get("type") or item.get("category") or item.get("categories")
        text = item.get("error") or item.get("description") or item.get("explanation") or item.get("text") or item.get("message")
        if text is None and "errors" in item:
            return " ".join(_flatten_error_texts(item["errors"]))
        if category and text:
            return f"{category}: {text}".strip()
        if text is not None:
            return str(text).strip()
        return " ".join(str(part).strip() for part in item.values() if part is not None).strip()
    return str(item).strip()


def _is_meaningful_error(text):
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return normalized not in _EMPTY_ERROR_TEXTS


def _coerce_to_items(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        if "categories" in value:
            return _coerce_to_items(value["categories"])
        if "category" in value:
            return _coerce_to_items(value["category"])
        return list(value.values())
    if isinstance(value, str):
        parsed = _try_parse_json(value)
        if parsed is not value:
            return _coerce_to_items(parsed)
        return re.split(r"[,;/\n]+|\band\b", value)
    return [value]


def _try_parse_json(value):
    text = value.strip()
    if not text or text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _normalize_key(value):
    return re.sub(r"\s+", " ", value.strip().lower().replace("-", "_").replace("_", " "))
