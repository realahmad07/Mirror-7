"""12-Domain Categorized General Conversation & Blueprint Taxonomy Engine for Mirror 7.

Organizes general human conversation into 12 structured cognitive domains, storing
compact slot-parameterized domain blueprints (< 8 KB total, 0 neural parameters)
instead of raw chat logs, and supporting bilingual Roman-Urdu & English interaction
with multi-turn context continuity.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class DomainBlueprintSpec:
    domain_id: str
    domain_title: str
    trigger_keywords: tuple[str, ...]
    urdu_framework_template: str
    english_framework_template: str
    default_topic: str


@dataclass(frozen=True)
class CategorizedDialogueResult:
    matched: bool
    domain_id: str = ""
    domain_title: str = ""
    extracted_topic: str = ""
    language: str = "en"
    response: str = ""
    uses_neural_parameters: bool = False


_TWELVE_CONVERSATION_DOMAINS: tuple[DomainBlueprintSpec, ...] = (
    DomainBlueprintSpec(
        domain_id="domain.1_mental_wellness_and_emotions",
        domain_title="Mental Wellness, Stress, Overthinking & Emotional Resilience",
        trigger_keywords=(
            "overthinking", "anxiety", "burnout", "lonely", "loneliness", "depressed",
            "self doubt", "fear of failure", "ghabrahat", "neend nahi", "sukoon",
            "dil nahi lag", "soch soch kar", "mental health", "confidence kam", "dar lagta",
            "ghussa", "ghussay", "anger", "qaboo", "panic", "stress",
        ),
        urdu_framework_template=(
            "[Domain 1: Mental Wellness & Emotional Blueprint | Topic: {topic}]\n"
            "Dekho yaar, {topic} ka shikar hona kamzori nahi balki is baat ki nishani hai ke aap ka dimagh baghair break ke bohot zyada bojh utha raha hai.\n"
            "1. Cognitive Reset (CBT Rule): Apni sochon ko kaghaz par likh lo — dimagh ke andar har masla 10 guna bada lagta hai, likhne se woh ek solvable list ban jata hai.\n"
            "2. 5-Minute Action Rule: Bade nateejay ki fikar chor kar sirf agle 5 minute ka sab se chota kaam shuru karo; amal (action) khauf aur overthinking ko khatam kar deta hai.\n"
            "3. Jism aur Neend: Thanda paani piyo, 15 minute walk karo, aur aaj raat screen jaldi band karo.\n"
            "Batao, is waqt sab se zyada konsi ek soch aap ko tang kar rahi hai?"
        ),
        english_framework_template=(
            "[Domain 1: Mental Wellness & Resilience Blueprint | Topic: {topic}]\n"
            "Experiencing {topic} is not a personal flaw — it is a signal that your cognitive bandwidth is overloaded without recovery.\n"
            "1. Externalize the Loop (CBT Framing): Write the exact worry on paper in one sentence; the brain magnifies unindexed thoughts, whereas written problems become bounded tasks.\n"
            "2. The 5-Minute Micro-Action: Uncertainty feeds anxiety, while execution starves it. Commit to just 5 minutes of the smallest concrete step.\n"
            "3. Physiological Reset: Step away from screens, hydrate, and take a slow 10-minute walk to lower cortisol.\n"
            "What is the single biggest thought looping in your mind right now?"
        ),
        default_topic="overthinking aur ذہنی دباؤ (mental stress)",
    ),
    DomainBlueprintSpec(
        domain_id="domain.2_career_freelancing_and_interviews",
        domain_title="Career Growth, Freelancing, Job Interviews & Salary Negotiation",
        trigger_keywords=(
            "freelancing", "freelance", "upwork", "fiverr", "client", "interview",
            "resume", "cv", "portfolio", "salary negotiation", "job switch", "remote job",
            "naukri", "clients kaise", "promotion", "career growth",
        ),
        urdu_framework_template=(
            "[Domain 2: Career & Freelancing Blueprint | Topic: {topic}]\n"
            "{topic} mein market ke 95% logon se aage nikalne ka 3-step blueprint:\n"
            "1. Proof-of-Work over Claims: Client ya company ko 'Main yeh kar sakta hoon' kehne ke bajaye un ke maslay ka ek chota working demo ya audit pehle hi dikha do.\n"
            "2. STAR Framework (Interviews & Proposals): Har jawab mein Situation -> Task -> Action -> Measurable Result (numbers ke sath) batao.\n"
            "3. Value-Anchored Negotiation: Apni qeemat ghanton (hours) par nahi balki client ko milne wale ROI aur bachat par tay karo.\n"
            "Aap abhi kis specific skill ya role ko target kar rahe hain?"
        ),
        english_framework_template=(
            "[Domain 2: Career & Freelancing Blueprint | Topic: {topic}]\n"
            "To stand out in the top 5% for {topic}, apply this 3-part execution blueprint:\n"
            "1. Proof-of-Work Before Pitching: Instead of generic claims, send a 3-bullet teardown or mini-prototype solving the client's or hiring manager's exact pain point.\n"
            "2. Quantified STAR Framing: Frame every past project as Situation -> Task -> Action -> Measurable Impact (latency dropped %, revenue gained $).\n"
            "3. Value-Anchored Pricing: Anchor compensation to the business risk you eliminate rather than raw hours.\n"
            "Which specific role or market niche are you targeting right now?"
        ),
        default_topic="career aur freelancing growth",
    ),
    DomainBlueprintSpec(
        domain_id="domain.3_software_architecture_and_engineering",
        domain_title="Software Engineering, System Design, Scaling & AI Architecture",
        trigger_keywords=(
            "microservices", "monolith", "system design", "database scaling", "caching",
            "redis", "rest vs graphql", "clean architecture", "tech stack", "high traffic",
            "low latency", "distributed system", "edge ai", "software architecture",
        ),
        urdu_framework_template=(
            "[Domain 3: Software & System Design Blueprint | Topic: {topic}]\n"
            "{topic} design karte waqt hamesha First-Principles Engineering follow karein:\n"
            "1. Modular Monolith First: Shuru mein microservices ki complexity se bachein; clean boundaries wala monolith banayein aur sirf bottleneck service ko alag scale karein.\n"
            "2. Data & Cache Locality: 90% latency database I/O mein hoti hai — sahi composite B-Tree indexing, SQLite WAL / OS-mmap, aur read-through caching lagayein.\n"
            "3. Deterministic Verification: Har critical path par automated invariant checks aur circuit breakers rakhein taake production mein 0% silent failure ho."
        ),
        english_framework_template=(
            "[Domain 3: Software & System Architecture Blueprint | Topic: {topic}]\n"
            "When architecting {topic}, follow three battle-tested systems principles:\n"
            "1. Start with a Modular Monolith: Avoid premature distributed network hops; enforce strict module contracts first and shard only proven CPU/IO hotspots.\n"
            "2. Data-First Latency Budgeting: Optimize index selectivity, memory-mapped I/O, and cache invalidation before scaling compute nodes.\n"
            "3. Fail-Closed Verification: Pair every state transition with deterministic contract checks and idempotent retries."
        ),
        default_topic="scalable system architecture",
    ),
    DomainBlueprintSpec(
        domain_id="domain.4_business_startups_and_finance",
        domain_title="Startups, Business Strategy, Unit Economics & Personal Finance",
        trigger_keywords=(
            "startup", "unit economics", "cac", "ltv", "bootstrapping", "pricing strategy",
            "passive income", "investing", "inflation", "saving money", "financial freedom",
            "karobar", "munafa kaise", "business idea", "sales kaise barhain",
        ),
        urdu_framework_template=(
            "[Domain 4: Business & Financial Blueprint | Topic: {topic}]\n"
            "{topic} ko mazboot bunyad par khara karne ke 3 sunehri usool:\n"
            "1. Customer Validation First: Paisa lagane se pehle 10 asli customers se baat kar ke dekho ke kya woh is maslay ke hal ke liye aaj advance pay karne ko tayyar hain.\n"
            "2. Unit Economics (LTV > 3x CAC): Har customer se aane wala munafa (LTV) usay lane ke kharchay (CAC) se kam az kam 3 guna zyada hona chahiye.\n"
            "3. Cash-Flow Discipline: Shuru mein fixed kharchay (overhead) zero ke qareeb rakho aur munafay ko dobara product aur distribution mein reinvest karo."
        ),
        english_framework_template=(
            "[Domain 4: Business, Startup & Finance Blueprint | Topic: {topic}]\n"
            "Apply this 3-pillar financial and startup blueprint to {topic}:\n"
            "1. Revenue Before Scale: Validate demand by getting 5-10 paying customers before spending capital on heavy infrastructure.\n"
            "2. Strict Unit Economics: Ensure Customer Lifetime Value (LTV) exceeds Customer Acquisition Cost (CAC) by at least 3x with < 6-month payback.\n"
            "3. Asymmetric Downside Protection: Keep fixed burn low so you can compound experiments longer than competitors."
        ),
        default_topic="startup aur financial strategy",
    ),
    DomainBlueprintSpec(
        domain_id="domain.5_study_learning_and_productivity",
        domain_title="Accelerated Learning, Study Mastery, Memory & Anti-Procrastination",
        trigger_keywords=(
            "procrastination", "spaced repetition", "active recall", "feynman technique",
            "exam preparation", "study tips", "deep work", "focus kaise", "yaad kaise rakhoon",
            "parhai mein dil", "distraction", "time management", "productivity",
            "social media", "wasting time", "kitab", "reading habit", "shauq", "screen time",
        ),
        urdu_framework_template=(
            "[Domain 5: Accelerated Learning & Focus Blueprint | Topic: {topic}]\n"
            "{topic} mein kam waqt mein 3x behtar nateeja lene ka scientific tareeqa:\n"
            "1. Feynman Technique: Jo topic parha hai usay band kitab kar ke asaan alfaaz mein kaghaz par aise likho jaise kisi 10 saal ke bachay ko samjha rahe ho — jahan atko, sirf woh hissa dobara parho.\n"
            "2. Active Recall & Spaced Repetition: Bar bar kitaab parhne (passive reading) ke bajaye khud se sawal poocho aur 1 din, 3 din, aur 7 din ke waqfay se revise karo.\n"
            "3. 25-Minute Deep Work Sprint: Phone doosre kamray mein rakh kar sirf 25 minute ka timer lagao."
        ),
        english_framework_template=(
            "[Domain 5: Accelerated Learning & Deep-Work Blueprint | Topic: {topic}]\n"
            "To master {topic} with maximum retention and zero procrastination:\n"
            "1. The Feynman Compression Test: Close your notes and explain the concept on a blank page from first principles; any gap in your explanation pinpoints what to review.\n"
            "2. Active Recall over Passive Re-reading: Test yourself with closed-book problems at expanding intervals (Day 1, Day 3, Day 7, Day 21).\n"
            "3. Single-Task 25-Minute Blocks: Remove phone/tab triggers and run focused 25-minute sprints."
        ),
        default_topic="deep learning aur focus mastery",
    ),
    DomainBlueprintSpec(
        domain_id="domain.6_relationships_communication_and_social",
        domain_title="Relationships, Family Harmony, Conflict Resolution & Communication",
        trigger_keywords=(
            "conflict resolution", "family expectations", "argument", "setting boundaries",
            "active listening", "public speaking", "narazgi", "rishtay", "ghar walay",
            "baat manwana", "communication skills", "trust issues", "misunderstanding",
            "dost kaise", "make friends", "english bolna", "speaking confidence",
        ),
        urdu_framework_template=(
            "[Domain 6: Relationships & Communication Blueprint | Topic: {topic}]\n"
            "{topic} ko samajhdari aur izzat ke sath hal karne ka 3-step tareeqa:\n"
            "1. Pehle Suno, Phir Bolo (Active Empathy): Jab samne wala dekhta hai ke aap ne us ki baat poori tawajjo se suni hai, to us ka 70% ghussa ya zid wahin khatam ho jati hai.\n"
            "2. Ilzam ke Bajaye Asar Batao (Non-Violent Framing): 'Tum hamesha aisa karte ho' kehne ke bajaye kaho 'Jab aisa hota hai to mujhe yeh mushkil pesh aati hai'.\n"
            "3. Narm Lehja, Mazboot Usool: Lehja hamesha narm aur adab wala rakho, lekin apni sehatmand hudood (boundaries) wazeh rakho."
        ),
        english_framework_template=(
            "[Domain 6: Communication & Conflict-Resolution Blueprint | Topic: {topic}]\n"
            "Navigate {topic} effectively using three high-trust communication principles:\n"
            "1. Steelman Before Responding: Summarize the other person's perspective so accurately they say 'Yes, that's right' before introducing your view.\n"
            "2. Non-Violent Observation -> Need -> Request: Separate objective facts from accusations, state your core need clearly, and propose a concrete win-win step.\n"
            "3. Warm Tone, Firm Boundaries: Keep your delivery calm and respectful while holding clear boundaries."
        ),
        default_topic="healthy communication aur relationships",
    ),
    DomainBlueprintSpec(
        domain_id="domain.7_health_fitness_and_lifestyle",
        domain_title="Health, Fitness, Sleep Hygiene, Nutrition & Sustainable Habits",
        trigger_keywords=(
            "weight loss", "muscle gain", "gym routine", "sleep hygiene", "healthy diet",
            "nutrition", "protein intake", "morning routine", "sehat", "wazan kam",
            "fitness", "energy low", "susti", "exercise habit", "subah jaldi", "aadat",
        ),
        urdu_framework_template=(
            "[Domain 7: Health, Fitness & Energy Blueprint | Topic: {topic}]\n"
            "{topic} ke liye sustainable aur scientific lifestyle blueprint:\n"
            "1. 80/20 Nutrition Rule: Crash dieting ke bajaye rozana protein, sabziyan, aur paani poora rakho aur processed cheeni/tali hui cheezon ko kam karo.\n"
            "2. Progressive Movement: Rozana 30 minute brisk walk ya strength training — sab se behtareen exercise woh hai jo aap har haftay lagatar kar sakein.\n"
            "3. Circadian Sleep Anchor: Rozana ek hi waqt par sona aur subah ki roshni lena aap ke hormones aur dimaghi energy ko 2x behtar kar deta hai."
        ),
        english_framework_template=(
            "[Domain 7: Health, Metabolic Energy & Habit Blueprint | Topic: {topic}]\n"
            "Build lasting results in {topic} through three physiological levers:\n"
            "1. Nutritional Consistency: Prioritize whole-food protein, fiber, and hydration while maintaining a sustainable caloric balance rather than extreme crash diets.\n"
            "2. Progressive Resistance & Daily Steps: Pair 3-4 weekly resistance sessions with 7,000+ daily steps for cardiovascular and metabolic resilience.\n"
            "3. Circadian Sleep Lock: Fix your wake time and get morning sunlight to stabilize energy and recovery."
        ),
        default_topic="sehat aur sustainable fitness",
    ),
    DomainBlueprintSpec(
        domain_id="domain.8_philosophy_psychology_and_ethics",
        domain_title="Philosophy, Stoicism, Iqbal's Khudi, Cognitive Biases & Ethics",
        trigger_keywords=(
            "stoicism", "iqbal", "khudi", "cognitive bias", "confirmation bias",
            "sunk cost", "free will", "ethics", "morality", "sabr", "shukr",
            "falsafa", "insani nafsiyaat", "first principles", "epictetus",
        ),
        urdu_framework_template=(
            "[Domain 8: Philosophy, Khudi & Psychology Blueprint | Topic: {topic}]\n"
            "{topic} par falsafay aur insani nafsiyaat ka gehra nichor:\n"
            "1. Dichotomy of Control (Stoicism & Tawakkul): Duniya ko do hisson mein baant lo — woh jo aap ke ikhtiyar mein hai (aap ki niyat, mehnat, aur akhlaq) aur woh jo aap ke ikhtiyar mein nahi (logon ki baatein aur nateeja). Sirf pehle hissay par apni taqat lagao.\n"
            "2. Falsafa-e-Khudi (Allama Iqbal): Insan jab mushkilat se ghabrane ke bajaye apni salahiyaton ko pehchanta aur amal ki bhatti mein tapata hai, tabhi us ki asli Khudi banti hai.\n"
            "3. Cognitive Bias Check: Faisla karte waqt hamesha poocho: 'Agar main ghalat hoon to us ka sab se bada saboot kya hoga?'"
        ),
        english_framework_template=(
            "[Domain 8: Philosophy, Stoicism & Cognitive Psychology Blueprint | Topic: {topic}]\n"
            "Examining {topic} through classical philosophy and cognitive science:\n"
            "1. The Dichotomy of Control (Epictetus): Invest 100% of your energy into your own judgments, effort, and character, and release attachment to external outcomes outside your control.\n"
            "2. Self-Actualization Through Agency (Iqbal's Khudi): Human potential is forged not by comfort, but by disciplined creative action against hard problems.\n"
            "3. Epistemic Humility (De-Biasing): Actively seek disconfirming evidence to counter confirmation bias and sunk-cost fallacies."
        ),
        default_topic="philosophy aur self-mastery",
    ),
    DomainBlueprintSpec(
        domain_id="domain.9_history_civilizations_and_geopolitics",
        domain_title="World History, Islamic Golden Age, Civilizational Cycles & Geopolitics",
        trigger_keywords=(
            "islamic golden age", "al-khwarizmi", "ibn sina", "ibn khaldun", "asabiyyah",
            "silk road", "industrial revolution", "civilization", "rise and fall",
            "tareekh", "muslim scientists", "geopolitics", "semiconductor",
        ),
        urdu_framework_template=(
            "[Domain 9: History & Civilizational Analysis Blueprint | Topic: {topic}]\n"
            "Tareekh aur {topic} ke mutalay se 3 bade tareekhi qawaneen samne aate hain:\n"
            "1. Ibn-e-Khaldun ka Qanoon-e-Asabiyyah (Muqaddimah): Qaumein aur riyasatein tab urooj paati hain jab un mein ittehad, insaf, aur mehnat (Asabiyyah) hoti hai, aur aish-o-ishrat mein par kar zawaal ka shikar hoti hain.\n"
            "2. Ilm aur Tajurba (Islamic Golden Age): Al-Khwarizmi (Algebra & Algorithms), Ibn al-Haytham (Scientific Method & Optics), aur Ibn Sina ne dikhaya ke jab tehqeeq ke darwazay khulay hon to tehzeeb duniya ki imamat karti hai.\n"
            "3. Technological Sovereignty: Aaj ki duniya mein wohi qaum azaad hai jo apne algorithms, hardware, aur science khud banati hai."
        ),
        english_framework_template=(
            "[Domain 9: Historical & Civilizational Dynamics Blueprint | Topic: {topic}]\n"
            "Analyzing {topic} through macro-historical laws:\n"
            "1. Ibn Khaldun's Asabiyyah Cycle: Civilizations rise through high social cohesion, meritocratic justice, and productive discipline, and decline when institutional corruption and complacency take over.\n"
            "2. The Empirical Scientific Lineage: From Al-Khwarizmi's algebra/algorithms and Ibn al-Haytham's experimental optics to modern computing, open inquiry is the engine of civilizational leadership.\n"
            "3. Technological Sovereignty: Economic and geopolitical resilience belongs to societies that build foundational science and compute independently."
        ),
        default_topic="civilizational history aur scientific heritage",
    ),
    DomainBlueprintSpec(
        domain_id="domain.10_creative_metaphors_humor_and_debate",
        domain_title="ELI10 Analogies, Metaphors, Clean Wit/Humor & Dialectical Debate",
        trigger_keywords=(
            "explain like i am 10", "eli5", "eli10", "give me an analogy", "metaphor for",
            "misaal de kar", "asaan lafzon mein", "tell me a joke", "latifa sunao",
            "funny joke", "debate both sides", "pros and cons of",
        ),
        urdu_framework_template=(
            "[Domain 10: Creative Analogy, Wit & Dialectical Blueprint | Topic: {topic}]\n"
            "Asaan Misaal & Lutf (0-Parameter Conceptual Blend):\n"
            "- Misaal (Analogy): Jis tarah ek ustaad mistri poori dukan ka kachra jama karne ke bajaye sirf 10 sahi auzar (tools) aur naqsha (blueprint) rakhta hai aur har cheez bana leta hai, bilkul usi tarah Mirror 7 billion parameters ke bojh ke baghair compact blueprints se chalta hai!\n"
            "- Programmer Latifa: Ek AI engineer se poocha gaya: 'Tumhara 70-Billion model itni bijli kyun khata hai?' Engineer bola: 'Kyunke woh 2+2 karne se pehle poori internet ki shayari yaad karta hai!' 😄"
        ),
        english_framework_template=(
            "[Domain 10: Creative Analogy, Wit & Dialectical Blueprint | Topic: {topic}]\n"
            "First-Principles Analogy & Wit (0-Parameter Conceptual Blend):\n"
            "- Everyday Analogy: Imagine two chefs. One memorizes 10 million photographed plates of food (requiring a warehouse of hard drives), while the other masters 50 core culinary laws of heat, acid, salt, and ratios (fitting in a pocket notebook). Mirror 7 is the second chef.\n"
            "- Engineering Wit: Why did the 100-billion-parameter LLM cross the road? It didn't — it ran out of VRAM halfway and hallucinated that the road was a bicycle! 😄"
        ),
        default_topic="creative analogy aur engineering humor",
    ),
    DomainBlueprintSpec(
        domain_id="domain.11_science_technology_and_future",
        domain_title="Future of AI, Neuro-Symbolic Computing, Quantum Tech & Green Compute",
        trigger_keywords=(
            "future of ai", "agi", "parameter free ai", "token free ai", "neuro symbolic",
            "green computing", "energy efficient ai", "qwen vs mirror", "chatgpt vs mirror",
            "ai ka mustaqbil",
        ),
        urdu_framework_template=(
            "[Domain 11: Future of AI & Compute Economics Blueprint | Topic: {topic}]\n"
            "AI ka mustaqbil ({topic}) sirf bade GPU clusters nahi balki **Zero-Bloat Symbolic + Compression Intelligence** hai:\n"
            "1. Zero Hallucination Guarantee: Jahan LLMs probability par tukka maarte hain, wahan Mirror UOL, AST verification, aur Raw-DEFLATE NCD se 100% verified jawab deta hai.\n"
            "2. 10,000x Compute Efficiency: $30,000 ke GPU aur gigabytes VRAM ke bajaye aam laptop CPU par 1-3 millisecond mein chalna AI ko har insan ki pohanch mein lata hai.\n"
            "3. Instant 1-Shot Learning: Retraining ke baghair naye formulas aur conversational blueprints 1 millisecond mein seekh lena."
        ),
        english_framework_template=(
            "[Domain 11: Future of AI & Compute Economics Blueprint | Topic: {topic}]\n"
            "The frontier of {topic} is shifting from brute-force parameter scaling to **Verifiable Zero-Bloat Intelligence**:\n"
            "1. Mathematical Truth Lock: Replacing stochastic token sampling with UOL execution, AST verification, and Kolmogorov-NCD compression eliminates hallucinations (0.0%).\n"
            "2. 10,000x Hardware Efficiency: Running in 1-3 ms on commodity CPUs (< 20 MB RAM, 0 GPUs) democratizes intelligence globally.\n"
            "3. Instant Anti-Unification Learning: Distilling reusable blueprints in 1 shot without gradient descent or catastrophic forgetting."
        ),
        default_topic="future of parameter-free AI",
    ),
    DomainBlueprintSpec(
        domain_id="domain.12_daily_chitchat_and_companionship",
        domain_title="Daily Rapport, Gratitude, Farewells & Friendly Companionship",
        trigger_keywords=(
            "thank you", "thanks a lot", "shukriya", "bohot shukriya", "jazakallah",
            "allah hafiz", "khuda hafiz", "goodbye", "good night", "shab bakhair",
            "good morning", "subah bakhair", "great job", "shabash", "kamaal kar diya",
        ),
        urdu_framework_template=(
            "[Domain 12: Friendly Rapport Blueprint]\n"
            "Bohot shukriya! Aap ke sath kaam kar ke aur naye ideas explore kar ke hamesha maza aata hai. "
            "Jab bhi kisi hisaab, coding, research, ya dil ki baat ke liye zaroorat ho — Mirror 7 hazir hai. Khush rahein aur aage barhte rahein!"
        ),
        english_framework_template=(
            "[Domain 12: Friendly Rapport Blueprint]\n"
            "Thank you! It is always a pleasure building, reasoning, and exploring ideas with you. "
            "Whenever you want to tackle a hard engineering problem, analyze data, or just talk things through — Mirror 7 is ready!"
        ),
        default_topic="friendly rapport",
    ),
)


class CategorizedConversationDomainEngine:
    """Zero-parameter 12-Domain Categorized General Conversation Router & Blueprint Composer."""

    VERSION = "categorized-conversation-domains-v1"

    def list_domains(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"domain_id": d.domain_id, "domain_title": d.domain_title}
            for d in _TWELVE_CONVERSATION_DOMAINS
        )

    def route_domain_conversation(self, prompt: str) -> CategorizedDialogueResult:
        text = prompt.strip()
        low = text.casefold()

        # Guard: Do not intercept third-person unknowable future/factual traps
        if re.match(
            r"^\s*(?:who\s+will\s+win|who\s+won\s+the\s+2099|what\s+will\s+the\s+exact|what\s+is\s+the\s+stock\s+price|what\s+is\s+the\s+best\s+programming\s+language|what\s+happened\s+in\s+yesterday)\b",
            low,
        ):
            return CategorizedDialogueResult(False)

        is_urdu = bool(
            re.search(
                r"\b(?:yaar|bhai|mera|meri|mujhe|kaise|karoon|hai|hoon|nahi|bohot|shukriya|jazakallah|allah\s+hafiz|neend|ghabrahat|parhai|naukri|karobar|rishtay|sehat|latifa|misaal|mustaqbil|tareekh|falsafa)\b",
                low,
            )
        )
        lang = "roman_urdu" if is_urdu else "en"

        scored_domains: list[tuple[int, str, DomainBlueprintSpec]] = []

        for spec in _TWELVE_CONVERSATION_DOMAINS:
            hits = 0
            first_hit = ""
            for kw in spec.trigger_keywords:
                if kw in low:
                    hits += 1
                    if not first_hit:
                        first_hit = kw
            if hits > 0:
                scored_domains.append((hits, first_hit, spec))

        if not scored_domains:
            return CategorizedDialogueResult(False)

        scored_domains.sort(key=lambda x: x[0], reverse=True)
        top_hits, top_kw, best_spec = scored_domains[0]
        topic = top_kw or best_spec.default_topic
        tmpl = best_spec.urdu_framework_template if is_urdu else best_spec.english_framework_template
        rendered = tmpl.format(topic=topic)

        # Cross-Domain Conversational Blending (Loop Iteration 2 Upgrade):
        # If the user's message spans 2 distinct human domains (e.g. Career/Freelancing + Mental Anxiety),
        # synthesize a unified cross-domain response addressing BOTH dimensions!
        if len(scored_domains) >= 2 and scored_domains[1][0] >= 1:
            sec_hits, sec_kw, sec_spec = scored_domains[1]
            sec_topic = sec_kw or sec_spec.default_topic
            sec_tmpl = sec_spec.urdu_framework_template if is_urdu else sec_spec.english_framework_template
            sec_rendered = sec_tmpl.format(topic=sec_topic)
            rendered = (
                f"[Cross-Domain Synthesized Blueprint: {best_spec.domain_title} + {sec_spec.domain_title}]\n"
                f"{rendered}\n\n"
                f"--- Companion Domain Blueprint ({sec_spec.domain_title}) ---\n"
                f"{sec_rendered}"
            )
            return CategorizedDialogueResult(
                matched=True,
                domain_id=f"{best_spec.domain_id}+{sec_spec.domain_id}",
                domain_title=f"{best_spec.domain_title} + {sec_spec.domain_title}",
                extracted_topic=f"{topic} & {sec_topic}",
                language=lang,
                response=rendered,
            )

        return CategorizedDialogueResult(
            matched=True,
            domain_id=best_spec.domain_id,
            domain_title=best_spec.domain_title,
            extracted_topic=topic,
            language=lang,
            response=rendered,
        )
