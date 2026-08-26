"""Bilingual (Arabic / English) skills taxonomy.

A seed of ~100 canonical skills with their Arabic and English aliases ships in
code; the DB table ``skills_taxonomy`` is seeded from it and can be extended at
runtime without a code change.  Normalisation folds an extracted skill string
onto its canonical form so "PostgreSQL", "postgres" and "قواعد بيانات
بوستجريس" all score as the same skill.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

_ARABIC_NOISE = re.compile(r"[ـً-ٰٟ]")
_PUNCT = re.compile(r"[^\w+#./ ؀-ۿ]+")


def canonicalise_text(value: str) -> str:
    out = unicodedata.normalize("NFKC", value or "").casefold()
    out = _ARABIC_NOISE.sub("", out)
    out = out.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    out = _PUNCT.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class SkillEntry:
    canonical: str
    aliases: list[str]
    category: str = "general"

    def as_dict(self) -> dict[str, Any]:
        return {"canonical": self.canonical, "aliases": self.aliases, "category": self.category}


def _e(canonical: str, category: str, *aliases: str) -> SkillEntry:
    return SkillEntry(canonical, list(aliases), category)


SEED_TAXONOMY: list[SkillEntry] = [
    # --- programming languages -------------------------------------------
    _e("python", "language", "python3", "بايثون", "بايثن"),
    _e("javascript", "language", "js", "ecmascript", "جافاسكريبت"),
    _e("typescript", "language", "ts", "تايب سكريبت"),
    _e("java", "language", "جافا"),
    _e("c#", "language", "csharp", "c sharp", "سي شارب"),
    _e("c++", "language", "cpp", "سي بلس بلس"),
    _e("go", "language", "golang", "جو"),
    _e("rust", "language", "رست"),
    _e("php", "language", "بي اتش بي"),
    _e("ruby", "language", "روبي"),
    _e("kotlin", "language", "كوتلن"),
    _e("swift", "language", "سويفت"),
    _e("scala", "language", "سكالا"),
    _e("r", "language", "r language", "لغة ار"),
    _e("sql", "language", "structured query language", "اس كيو ال"),
    _e("bash", "language", "shell scripting", "شل", "باش"),
    # --- backend frameworks ------------------------------------------------
    _e("django", "framework", "دجانغو", "جانغو"),
    _e("fastapi", "framework", "fast api", "فاست ابي"),
    _e("flask", "framework", "فلاسك"),
    _e("spring boot", "framework", "spring", "سبرينج"),
    _e("express", "framework", "expressjs", "اكسبرس"),
    _e("dotnet", "framework", ".net", "asp.net", "دوت نت"),
    _e("laravel", "framework", "لارافيل"),
    _e("rails", "framework", "ruby on rails", "ريلز"),
    _e("celery", "framework", "سيليري"),
    _e("graphql", "framework", "جراف كيو ال"),
    _e("rest api", "framework", "rest", "restful", "واجهات برمجية", "ريست"),
    _e("grpc", "framework", "جي ار بي سي"),
    _e("microservices", "architecture", "micro services", "الخدمات المصغرة", "خدمات مصغرة"),
    _e("event driven architecture", "architecture", "event-driven", "معمارية الأحداث"),
    _e("domain driven design", "architecture", "ddd", "التصميم الموجه بالمجال"),
    # --- frontend ----------------------------------------------------------
    _e("react", "frontend", "reactjs", "react.js", "رياكت"),
    _e("vue", "frontend", "vuejs", "فيو"),
    _e("angular", "frontend", "angularjs", "انجولار"),
    _e("next.js", "frontend", "nextjs", "نكست"),
    _e("html", "frontend", "html5", "اتش تي ام ال"),
    _e("css", "frontend", "css3", "sass", "scss", "سي اس اس"),
    _e("tailwind", "frontend", "tailwindcss", "تيلويند"),
    # --- data stores -------------------------------------------------------
    _e("postgresql", "database", "postgres", "psql", "بوستجريس", "بوستغريس"),
    _e("mysql", "database", "mariadb", "ماي سيكوال"),
    _e("sqlite", "database", "سكيو لايت"),
    _e("mongodb", "database", "mongo", "مونجو"),
    _e("redis", "database", "ريديس"),
    _e("elasticsearch", "database", "elastic", "opensearch", "الاستيك"),
    _e("cassandra", "database", "كاساندرا"),
    _e("dynamodb", "database", "دينامو"),
    _e("oracle", "database", "اوراكل"),
    _e("sql server", "database", "mssql", "اس كيو ال سيرفر"),
    # --- cloud / infra -----------------------------------------------------
    _e("aws", "cloud", "amazon web services", "امازون ويب سيرفيس", "اي دبليو اس"),
    _e("gcp", "cloud", "google cloud", "جوجل كلاود"),
    _e("azure", "cloud", "microsoft azure", "ازور"),
    _e("docker", "devops", "containers", "دوكر", "حاويات"),
    _e("kubernetes", "devops", "k8s", "كوبرنيتس"),
    _e("terraform", "devops", "تيرافورم"),
    _e("ansible", "devops", "انسيبل"),
    _e("jenkins", "devops", "جنكينز"),
    _e("ci/cd", "devops", "cicd", "continuous integration", "التكامل المستمر"),
    _e("github actions", "devops", "gh actions", "جيت هب اكشنز"),
    _e("linux", "devops", "unix", "لينكس"),
    _e("nginx", "devops", "انجن اكس"),
    _e("git", "devops", "version control", "جيت", "التحكم بالإصدارات"),
    _e("prometheus", "devops", "بروميثيوس"),
    _e("grafana", "devops", "جرافانا"),
    _e("observability", "devops", "monitoring", "المراقبة", "الرصد"),
    # --- data / analytics --------------------------------------------------
    _e("pandas", "data", "بانداز"),
    _e("numpy", "data", "نمباي"),
    _e("spark", "data", "apache spark", "pyspark", "سبارك"),
    _e("airflow", "data", "apache airflow", "ايرفلو"),
    _e("dbt", "data", "data build tool", "دي بي تي"),
    _e("kafka", "data", "apache kafka", "كافكا"),
    _e("etl", "data", "elt", "data pipelines", "خطوط البيانات", "تحويل البيانات"),
    _e("data warehousing", "data", "warehouse", "snowflake", "bigquery", "redshift", "مستودع البيانات"),
    _e("power bi", "analytics", "powerbi", "باور بي اي"),
    _e("tableau", "analytics", "تابلوه", "تابلو"),
    _e("looker", "analytics", "لوكر"),
    _e("excel", "analytics", "microsoft excel", "اكسل", "الاكسل"),
    _e("data visualization", "analytics", "dataviz", "تصور البيانات", "تصوير البيانات"),
    _e("statistics", "analytics", "statistical analysis", "الإحصاء", "التحليل الاحصائي"),
    _e("ab testing", "analytics", "a/b testing", "experimentation", "اختبار ا ب"),
    _e("data analysis", "analytics", "تحليل البيانات", "محلل بيانات"),
    _e("machine learning", "ml", "ml", "التعلم الالي", "تعلم الالة"),
    _e("deep learning", "ml", "neural networks", "التعلم العميق"),
    _e("nlp", "ml", "natural language processing", "معالجة اللغات الطبيعية"),
    _e("computer vision", "ml", "الرؤية الحاسوبية"),
    _e("mlops", "ml", "ام ال اوبس"),
    # --- practices ---------------------------------------------------------
    _e("agile", "practice", "scrum", "kanban", "اجايل", "سكرم"),
    _e("code review", "practice", "مراجعة الكود"),
    _e("unit testing", "practice", "pytest", "junit", "tdd", "اختبارات الوحدة"),
    _e("system design", "practice", "architecture design", "تصميم الأنظمة"),
    _e("performance tuning", "practice", "optimization", "تحسين الأداء"),
    _e("security", "practice", "appsec", "owasp", "الأمن السيبراني", "الأمان"),
    _e("technical writing", "practice", "documentation", "الكتابة التقنية", "التوثيق"),
    _e("mentoring", "practice", "coaching", "التوجيه", "الإرشاد"),
    _e("stakeholder management", "practice", "إدارة أصحاب المصلحة"),
    _e("project management", "practice", "إدارة المشاريع"),
    _e("communication", "soft", "التواصل", "مهارات التواصل"),
    _e("problem solving", "soft", "حل المشكلات"),
    _e("teamwork", "soft", "collaboration", "العمل الجماعي"),
    _e("leadership", "soft", "القيادة"),
    _e("arabic", "language-skill", "اللغة العربية", "عربي"),
    _e("english", "language-skill", "اللغة الإنجليزية", "انجليزي"),
]


class Taxonomy:
    """Alias -> canonical lookup, seeded from code and extensible from the DB."""

    def __init__(self, entries: Iterable[SkillEntry] | None = None) -> None:
        self._entries: dict[str, SkillEntry] = {}
        self._lookup: dict[str, str] = {}
        for entry in entries if entries is not None else SEED_TAXONOMY:
            self.add(entry)

    def add(self, entry: SkillEntry) -> None:
        self._entries[entry.canonical] = entry
        for token in [entry.canonical, *entry.aliases]:
            key = canonicalise_text(token)
            if key:
                self._lookup[key] = entry.canonical

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[SkillEntry]:
        return list(self._entries.values())

    def category_of(self, canonical: str) -> str:
        entry = self._entries.get(canonical)
        return entry.category if entry else "general"

    def normalise(self, skill: str) -> str:
        """Fold a raw skill string onto its canonical form.

        Unknown skills are returned in a canonicalised-but-unmapped form so they
        still compare consistently between a CV and a JD.
        """
        key = canonicalise_text(skill)
        if not key:
            return ""
        if key in self._lookup:
            return self._lookup[key]
        # Try the longest known alias contained in the string.
        best = ""
        for alias, canonical in self._lookup.items():
            if len(alias) >= 3 and alias in key and len(alias) > len(best):
                best = alias
        if best:
            return self._lookup[best]
        return key

    def normalise_all(self, skills: Iterable[str]) -> list[str]:
        seen: list[str] = []
        for skill in skills:
            value = self.normalise(skill)
            if value and value not in seen:
                seen.append(value)
        return seen

    # -- DB round-trip ------------------------------------------------------
    @classmethod
    def from_rows(cls, rows: Iterable[Any]) -> "Taxonomy":
        entries = [
            SkillEntry(row.canonical, json.loads(row.aliases_json or "[]"), row.category)
            for row in rows
        ]
        return cls(entries or SEED_TAXONOMY)


DEFAULT_TAXONOMY = Taxonomy()
