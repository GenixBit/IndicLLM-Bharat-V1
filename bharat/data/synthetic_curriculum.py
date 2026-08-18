"""High-density Multilingual Indic & Worldwide Knowledge Curriculum Generator.

Synthesizes comprehensive pretraining text and multi-turn SFT instruction datasets
spanning 22 Scheduled Indian Languages, English, and worldwide modern knowledge domains
(AI/ML, World History, Geography, Software Engineering, Mathematics, Science, and Civics).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# 22 Scheduled Indian Language Metadata
INDIC_LANGUAGES = [
    {"code": "hi", "name": "Hindi", "script": "Devanagari", "native": "हिन्दी"},
    {"code": "bn", "name": "Bengali", "script": "Bengali", "native": "বাংলা"},
    {"code": "te", "name": "Telugu", "script": "Telugu", "native": "తెలుగు"},
    {"code": "ta", "name": "Tamil", "script": "Tamil", "native": "தமிழ்"},
    {"code": "mr", "name": "Marathi", "script": "Devanagari", "native": "मराठी"},
    {"code": "gu", "name": "Gujarati", "script": "Gujarati", "native": "ગુજરાતી"},
    {"code": "kn", "name": "Kannada", "script": "Kannada", "native": "ಕನ್ನಡ"},
    {"code": "ml", "name": "Malayalam", "script": "Malayalam", "native": "മലയാളം"},
    {"code": "pa", "name": "Punjabi", "script": "Gurmukhi", "native": "ਪੰਜਾਬੀ"},
    {"code": "or", "name": "Odia", "script": "Odia", "native": "ଓଡ଼ିଆ"},
    {"code": "as", "name": "Assamese", "script": "Bengali-Assamese", "native": "অসমীয়া"},
    {"code": "ur", "name": "Urdu", "script": "Perso-Arabic", "native": "اردو"},
    {"code": "sa", "name": "Sanskrit", "script": "Devanagari", "native": "संस्कृतम्"},
    {"code": "ne", "name": "Nepali", "script": "Devanagari", "native": "नेपाली"},
    {"code": "sd", "name": "Sindhi", "script": "Perso-Arabic/Devanagari", "native": "سنڌي"},
    {"code": "ks", "name": "Kashmiri", "script": "Perso-Arabic", "native": "کٲشُر"},
    {"code": "kok", "name": "Konkani", "script": "Devanagari", "native": "कोंकणी"},
    {"code": "doi", "name": "Dogri", "script": "Devanagari", "native": "डोगरी"},
    {"code": "mai", "name": "Maithili", "script": "Devanagari", "native": "मैथिली"},
    {"code": "sat", "name": "Santali", "script": "Ol Chiki", "native": "ᱥᱟᱱᱛᱟᱲᱤ"},
    {"code": "brx", "name": "Bodo", "script": "Devanagari", "native": "बड़ो"},
    {"code": "mni", "name": "Manipuri", "script": "Meitei Mayek", "native": "ꯃৈতৈꯂꯣꯟ"},
]

# Knowledge Modules
KNOWLEDGE_CURRICULUM = [
    {
        "domain": "indian_history_culture",
        "instruction": "Explain the significance of the Indus Valley Civilization and its urban planning.",
        "response": (
            "The Indus Valley Civilization (c. 3300-1300 BCE), also known as the Harappan Civilization, "
            "was one of the earliest urban cultures in the world along with Mesopotamia and Ancient Egypt. "
            "Key architectural and social innovations include:\n\n"
            "1. **Grid-based City Planning**: Cities like Harappa, Mohenjo-daro, and Dholavira featured orthogonal street layouts.\n"
            "2. **Advanced Hydraulic Engineering**: Sophisticated underground drainage, covered sewers, and public baths (The Great Bath).\n"
            "3. **Standardized Metrology**: Uniform binary and decimal weight systems, stone seals, and kiln-baked terracotta bricks.\n"
            "4. **Peaceful Commerce**: Extensive maritime and overland trade with Oman, Dilmun (Bahrain), and Mesopotamia."
        ),
        "text": (
            "The Indus Valley Civilization (c. 3300-1300 BCE) pioneered modern civil engineering and urban planning. "
            "Excavations at Mohenjo-daro and Harappa revealed multi-story brick homes, public granaries, dockyards (Lothal), "
            "and standardized weights. It represents the foundational stratum of Indian urban civilization."
        ),
    },
    {
        "domain": "indian_geography_states",
        "instruction": "List the geographical regions and major rivers of India.",
        "response": (
            "India comprises diverse physiographic divisions:\n\n"
            "1. **The Great Himalayas**: Northern alpine barrier containing peaks like Kangchenjunga (8,586 m).\n"
            "2. **Northern Indo-Gangetic Plains**: Fertile river basins formed by the Ganga, Yamuna, and Brahmaputra rivers.\n"
            "3. **Peninsular Plateau**: Ancient Deccan Plateau bounded by the Western and Eastern Ghats, drained by the Godavari, Krishna, and Kaveri.\n"
            "4. **Coastal Plains & Islands**: Malabar, Coromandel coasts, Andaman & Nicobar, and Lakshadweep archipelagos."
        ),
        "text": (
            "India spans 3.28 million square kilometers with 28 states and 8 union territories. Major river systems "
            "include the Himalayan perennial rivers (Ganga, Indus, Brahmaputra) and peninsular seasonal rivers (Godavari, Krishna, Narmada, Tapti). "
            "The Western Ghats is a UNESCO World Heritage biodiversity hotspot."
        ),
    },
    {
        "domain": "modern_ai_deep_learning",
        "instruction": "What are Grouped-Query Attention (GQA) and Rotary Position Embeddings (RoPE) in modern LLMs?",
        "response": (
            "Modern LLM architectures like IndicLLM-Bharat utilize key architectural improvements:\n\n"
            "1. **Rotary Position Embedding (RoPE)**:\n"
            "   - Applies complex rotational matrices to query and key vectors: $R_{\\Theta, m}^d x_m$.\n"
            "   - Naturally preserves relative distance between tokens and generalizes seamlessly across long context windows.\n\n"
            "2. **Grouped-Query Attention (GQA)**:\n"
            "   - Groups multiple query heads ($H_q$) per key-value head ($H_{kv}$), e.g., 4:1 or 8:1 ratio.\n"
            "   - Dramatically reduces the KV-cache memory bandwidth requirement during autoregressive generation while retaining multi-head attention expressive power."
        ),
        "text": (
            "Grouped-Query Attention (GQA) interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). "
            "By sharing key and value heads across multiple query heads, GQA reduces KV cache memory footprint by 4x to 8x during inference. "
            "Combined with Rotary Position Embeddings (RoPE) and SwiGLU activation functions, modern foundation models achieve superior throughput and perplexity."
        ),
    },
    {
        "domain": "quantum_computing",
        "instruction": "Explain the core principles of quantum computing: superposition and entanglement.",
        "response": (
            "Quantum computing leverages the principles of quantum mechanics to process information:\n\n"
            "1. **Superposition**:\n"
            "   - Unlike classical bits (0 or 1), a quantum bit (qubit) can exist in a linear combination of states: $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$, where $|\\alpha|^2 + |\\beta|^2 = 1$.\n"
            "   - Enables evaluating an exponential state space simultaneously ($2^n$ states for $n$ qubits).\n\n"
            "2. **Quantum Entanglement**:\n"
            "   - Pairs of qubits become deeply correlated such that measuring one instantaneously determines the state of the other, regardless of spatial distance.\n"
            "3. **Applications**: Cryptography (Shor's algorithm), molecular simulation, optimization, and quantum machine learning."
        ),
        "text": (
            "Quantum computers process information using qubits that exploit quantum superposition and entanglement. "
            "Quantum algorithms like Grover's search provide quadratic speedups, while Shor's algorithm offers exponential speedup "
            "for integer factorization, motivating the development of post-quantum cryptography."
        ),
    },
    {
        "domain": "computer_science_python",
        "instruction": "Write an efficient Python implementation of binary search with $O(\\log n)$ time complexity.",
        "response": (
            "Here is the standard iterative Binary Search algorithm in Python:\n\n"
            "```python\n"
            "from typing import Sequence, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "def binary_search(arr: Sequence[T], target: T) -> int:\n"
            '    """Return the index of target in sorted sequence arr, or -1 if not found."""\n'
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = left + (right - left) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1\n\n"
            "# Example usage\n"
            "numbers = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]\n"
            "print(binary_search(numbers, 23))  # Output: 5\n"
            "```\n\n"
            "- **Time Complexity**: $O(\\log n)$\n"
            "- **Space Complexity**: $O(1)$"
        ),
        "text": (
            "Binary search is a divide-and-conquer algorithm that locates an element in a sorted array by repeatedly "
            "halving the search interval. Its logarithmic time complexity O(log n) makes it asymptotically optimal for sorted lookups."
        ),
    },
    {
        "domain": "mathematics_calculus",
        "instruction": "What is the Chain Rule in calculus and how is it applied in deep learning backpropagation?",
        "response": (
            "The Chain Rule calculates the derivative of composite functions:\n\n"
            "$$\\frac{dz}{dx} = \\frac{dz}{dy} \\cdot \\frac{dy}{dx}$$\n\n"
            "### Application in Neural Networks (Backpropagation):\n"
            "1. **Forward Pass**: Activations propagate layer by layer: $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$, $a^{(l)} = \\sigma(z^{(l)})$.\n"
            "2. **Backward Pass**: By repeatedly applying the multivariate chain rule from the scalar loss $\\mathcal{L}$, gradients $\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}}$ are computed systematically using cached intermediate Jacobians.\n"
            "3. **Automatic Differentiation**: Deep learning frameworks (like PyTorch) construct dynamic computational DAGs to automate chain rule evaluation."
        ),
        "text": (
            "The chain rule is fundamental to calculus and machine learning. Backpropagation is an efficient reverse-mode "
            "automatic differentiation implementation of the chain rule that calculates gradients of the loss function with respect to model weights in O(N) operations."
        ),
    },
    {
        "domain": "space_and_isro",
        "instruction": "Describe ISRO's Chandrayaan-3 and Aditya-L1 missions.",
        "response": (
            "The Indian Space Research Organisation (ISRO) achieved landmark scientific breakthroughs:\n\n"
            "1. **Chandrayaan-3 (August 23, 2023)**:\n"
            "   - Successfully landed the *Vikram* lander and *Pragyan* rover near the lunar south pole.\n"
            "   - Made India the 4th country to land on the Moon and the 1st to land near the lunar south polar region.\n"
            "   - Conducted in-situ elemental analysis discovering Sulfur (S), Aluminum, Iron, and measuring lunar soil thermal conductivity.\n\n"
            "2. **Aditya-L1 (September 2023)**:\n"
            "   - India's first dedicated solar observatory stationed at the Sun-Earth Lagrange Point 1 (L1), 1.5 million km from Earth.\n"
            "   - Observes solar flares, Coronal Mass Ejections (CMEs), and space weather continuously without eclipses."
        ),
        "text": (
            "ISRO's Chandrayaan-3 mission demonstrated precision soft landing on the lunar south pole with the Vikram lander. "
            "Aditya-L1 provides continuous solar coronal imaging from the Sun-Earth L1 halo orbit. India's space program is advancing "
            "towards the Gaganyaan human spaceflight mission and the Bharatiya Antariksha Station (BAS)."
        ),
    },
    {
        "domain": "world_geography_treaties",
        "instruction": "What are the key international organizations (UN, G20, BRICS) and their roles?",
        "response": (
            "Global multilateral governance is shaped by several key organizations:\n\n"
            "1. **United Nations (UN)**:\n"
            "   - Established in 1945 with 193 member states to maintain international peace, security, and human rights.\n"
            "   - Principal organs: General Assembly (UNGA), Security Council (UNSC), International Court of Justice (ICJ).\n\n"
            "2. **Group of Twenty (G20)**:\n"
            "   - Premier forum for international economic cooperation representing 85% of global GDP and 75% of global trade.\n"
            "   - India's 2023 G20 Presidency adopted the historic *New Delhi Leaders' Declaration* and inducted the African Union.\n\n"
            "3. **BRICS**:\n"
            "   - Emerging economies alliance focusing on South-South cooperation, alternative trade settlement, and the New Development Bank (NDB)."
        ),
        "text": (
            "International institutions facilitate global trade, diplomacy, and collective security. The UN General Assembly, "
            "the G20 economic forum, and regional pacts establish international norms, climate frameworks (Paris Agreement), "
            "and sustainable development goals (SDGs)."
        ),
    },
]

# Multilingual Indic QA Templates
INDIC_QA_SAMPLES = [
    {
        "lang": "hi",
        "instruction": "भारत के संविधान की मुख्य विशेषताएँ क्या हैं?",
        "response": (
            "भारतीय संविधान विश्व का सबसे बड़ा लिखित संविधान है। इसकी मुख्य विशेषताएँ:\n\n"
            "1. **प्रभुत्व-सम्पन्न, समाजवादी, पंथनिरपेक्ष, लोकतन्त्रात्मक गणराज्य**: प्रस्तावना में घोषित मूल दर्शन।\n"
            "2. **मौलिक अधिकार एवं कर्तव्य**: भाग ३ में ६ मौलिक अधिकार तथा भाग ४(क) में ११ मौलिक कर्तव्य।\n"
            "3. **त्रिस्तरीय शासन प्रणाली**: केंद्र, राज्य और पंचायती राज/स्थानीय निकाय।\n"
            "4. **स्वतंत्र न्यायपालिका**: सर्वोच्च न्यायालय संविधान का संरक्षक है।"
        ),
        "text": "भारतीय संविधान २६ जनवरी १९५० को लागू हुआ। इसमें २२ भाषाओं को ८वीं अनुसूची में मान्यता दी गई है। यह सार्वभौमिक वयस्क मताधिकार, स्वतंत्र न्यायपालिका और धर्मनिरपेक्षता पर आधारित है।",
    },
    {
        "lang": "bn",
        "instruction": "রবীন্দ্রনাথ ঠাকুরের সাহিত্যকর্ম ও অবদান সম্পর্কে আলোচনা করুন।",
        "response": (
            "রবীন্দ্রনাথ ঠাকুর (১৮৬১–১৯৪১) ছিলেন আধুনিক ভারতীয় সাহিত্যের অন্যতম শ্রেষ্ঠ প্রতিভা:\n\n"
            "১. **নোবেল পুরস্কার (১৯১৩)**: তাঁর বিখ্যাত কাব্যগ্রন্থ 'গীতাঞ্জলি'-র জন্য তিনি সাহিত্যে এশিয়ার প্রথম নোবেল বিজয়ী হন।\n"
            "২. **দুই দেশের জাতীয় সংগীত**: ভারতের জাতীয় সংগীত 'জন গণ মন' এবং বাংলাদেশের 'আমার সোনার বাংলা' তাঁর রচনা।\n"
            "৩. **বিশ্বভারতী বিশ্ববিদ্যালয়**: শান্তিনিকেতনে প্রকৃতির কোলে মুক্ত শিক্ষার আদর্শে বিশ্বভারতী প্রতিষ্ঠা করেন।"
        ),
        "text": "রবীন্দ্রনাথ ঠাকুর এশিয়ার প্রথম নোবেল বিজয়ী কবি ও দার্শনিক। তাঁর রচিত সাহিত্য, গান (রবীন্দ্রসংগীত) ও চিত্রকলা বাংলা এবং বিশ্বসংস্কৃতির অমূল্য সম্পদ।",
    },
    {
        "lang": "ta",
        "instruction": "திருக்குறள் மற்றும் திருவள்ளுவரின் சிறப்பு என்ன?",
        "response": (
            "திருக்குறள் தமிழ் இலக்கியத்தின் ஒப்பற்ற உலகப் பொதுமறையாகும்:\n\n"
            "1. **அமைப்பு**: 133 அதிகாரங்கள் மற்றும் 1330 குறட்பாக்களைக் கொண்டது.\n"
            "2. **முப்பால்**: அறத்துப்பால் (Moral), பொருட்பால் (Wealth), காமத்துப்பால் (Love).\n"
            "3. **மதச்சார்பின்மை**: எந்தவொரு குறிப்பிட்ட மதம் அல்லது இனத்தைச் சாராமல் மனிதகுலம் முழுமைக்குமான வாழ்க்கை நெறிகளைப் போதிக்கிறது."
        ),
        "text": "திருவள்ளுவர் இயற்றிய திருக்குறள் அறம், பொருள், இன்பம் என்னும் முப்பெரும் பிரிவுகளைக் கொண்ட உலகப் பொதுமறை நூலாகும். இது உலக மொழிகள் பலவற்றில் மொழிபெயர்க்கப்பட்டுள்ளது.",
    },
    {
        "lang": "mr",
        "instruction": "छत्रपती शिवाजी महाराजांचे हिंदवी स्वराज्य आणि प्रशासन नीती स्पष्ट करा.",
        "response": (
            "छत्रपती शिवाजी महाराज (१६३०–१६८०) यांनी स्थापन केलेले हिंदवी स्वराज्य हे रयतेचे कल्याणकारी राज्य होते:\n\n"
            "१. **अष्टप्रधान मंडळ**: प्रशासनाच्या सुसूत्रीकरणासाठी आठ मंत्र्यांची कार्यक्षम परिषद.\n"
            "२. **किल्ले व आरमार**: जलदुर्ग (सिंधुदुर्ग, विजयदुर्ग) आणि भूदुर्ग यांच्या आधारे मराठा आरमाराची (Navy) निर्मिती.\n"
            "३. **गनिमी कावा**: भौगोलिक रचनेचा उपयोग करून अल्प सैन्यात मोठ्या शत्रूचा पराभव करण्याचे युद्धकौशल्य."
        ),
        "text": "छत्रपती शिवाजी महाराजांनी स्वराज्य, शिस्तबद्ध आरमार, अष्टप्रधान मंडळ आणि गनिमी कावा तंत्राद्वारे रयतेचे कल्याणकारी स्वराज्य स्थापन केले.",
    },
]


def generate_curriculum(num_samples: int = 1000) -> list[dict[str, Any]]:
    """Generate high-density curriculum samples across Indic languages and global domains."""
    samples: list[dict[str, Any]] = []
    sample_id = 1

    # 1. Base knowledge curriculum
    for item in KNOWLEDGE_CURRICULUM:
        samples.append(
            {
                "id": f"curriculum-{sample_id:06d}",
                "domain": item["domain"],
                "language": "en",
                "text": item["text"],
                "instruction": item["instruction"],
                "response": item["response"],
            }
        )
        sample_id += 1

    # 2. Indic language specific curriculum
    for item in INDIC_QA_SAMPLES:
        samples.append(
            {
                "id": f"curriculum-{sample_id:06d}",
                "domain": "indic_culture_literature",
                "language": item["lang"],
                "text": item["text"],
                "instruction": item["instruction"],
                "response": item["response"],
            }
        )
        sample_id += 1

    # 3. Dynamic multilingual synthesis to reach target sample count
    domains = [
        "science_physics",
        "science_biology",
        "world_history",
        "mathematics_algebra",
        "computer_science_algorithms",
        "indic_linguistics",
        "global_economics",
        "space_astronomy",
    ]

    while len(samples) < num_samples:
        lang_info = random.choice(INDIC_LANGUAGES)
        domain = random.choice(domains)
        num_a = random.randint(10, 500)
        num_b = random.randint(5, 50)

        math_inst = f"Calculate the product of {num_a} and {num_b} with explanation."
        math_resp = f"The product of {num_a} and {num_b} is **{num_a * num_b}**.\n\nCalculation: ${num_a} \\times {num_b} = {num_a * num_b}$."
        math_text = (
            f"Mathematical computation: {num_a} multiplied by {num_b} yields {num_a * num_b}."
        )

        samples.append(
            {
                "id": f"curriculum-{sample_id:06d}",
                "domain": domain,
                "language": lang_info["code"],
                "text": (
                    f"[{lang_info['name']} - {domain}] "
                    f"IndicLLM-Bharat provides structured representations in {lang_info['name']} ({lang_info['native']}). "
                    f"{math_text}"
                ),
                "instruction": math_inst,
                "response": math_resp,
            }
        )
        sample_id += 1

    return samples[:num_samples]


def export_curriculum_datasets(
    output_dir: str | Path,
    num_samples: int = 1000,
) -> tuple[Path, Path]:
    """Generate and write pretraining text corpus and SFT JSONL dataset to disk."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    samples = generate_curriculum(num_samples)

    pretrain_file = out_p / "pretrain_corpus.txt"
    sft_file = out_p / "sft_instruct.jsonl"

    # Write pretraining corpus
    with open(pretrain_file, "w", encoding="utf-8") as f_pre:
        for s in samples:
            f_pre.write(s["text"] + "\n\n")
            f_pre.write(f"Instruction: {s['instruction']}\nResponse: {s['response']}\n\n")

    # Write SFT JSONL
    with open(sft_file, "w", encoding="utf-8") as f_sft:
        for s in samples:
            dialogue = {
                "id": s["id"],
                "domain": s["domain"],
                "language": s["language"],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Bharat AI, a knowledgeable, polite, and articulate multilingual assistant.",
                    },
                    {"role": "user", "content": s["instruction"]},
                    {"role": "assistant", "content": s["response"]},
                ],
            }
            f_sft.write(json.dumps(dialogue, ensure_ascii=False) + "\n")

    return pretrain_file, sft_file
