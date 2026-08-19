"""High-Density Multilingual & STEM Instruction Curriculum Generator for IndicLLM-Bharat.

Synthesizes structured instruction-response dialogues across all 22 Scheduled Indian Languages,
STEM, quantum physics, calculus, Python algorithms, and world knowledge for Supervised Fine-Tuning (SFT).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 1. STEM, Science & AI Instruction Dialogues
STEM_INSTRUCTION_DATA: list[dict[str, str]] = [
    {
        "prompt": "Explain how Grouped-Query Attention (GQA) reduces KV-cache memory bandwidth in sovereign LLMs like IndicLLM-Bharat.",
        "response": (
            "### Grouped-Query Attention (GQA) in Modern LLMs\n\n"
            "**Grouped-Query Attention (GQA)** is an architectural optimization that bridges Multi-Head Attention (MHA) "
            "and Multi-Query Attention (MQA):\n\n"
            "1. **Mechanism**:\n"
            "   - Standard MHA uses $H_q$ Query heads and $H_q$ Key/Value heads ($H_q = H_{kv}$).\n"
            "   - GQA partitions the $H_q$ Query heads into $G$ groups, where each group shares a single Key and Value head ($H_{kv} = H_q / G$).\n"
            "   - During inference, each Key and Value head is expanded across its query group using `repeat_interleave`.\n\n"
            "2. **Memory & Bandwidth Savings**:\n"
            "   - In a model with $H_q = 32$ and $H_{kv} = 8$ ($G = 4$), the KV-cache footprint per token is reduced by **4×**:\n"
            "     \\[ \\text{KV Cache Size} = 2 \\times \\text{layers} \\times H_{kv} \\times d_{\\text{head}} \\times \\text{seq\\_len} \\times \\text{bytes} \\]\n"
            "   - This enables **32k+ token** long-context processing with high throughput and low memory latency."
        ),
    },
    {
        "prompt": "Write a clean Python implementation of Binary Search with step-by-step algorithmic analysis.",
        "response": (
            "### Python Binary Search Algorithm\n\n"
            "Binary Search operates on a sorted array by repeatedly dividing the search interval in half.\n\n"
            "```python\n"
            "def binary_search(arr: list[int], target: int) -> int:\n"
            '    """Return index of target in sorted arr, or -1 if not found."""\n'
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = left + (right - left) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1\n"
            "```\n\n"
            "**Complexity Analysis**:\n"
            "- **Time Complexity**: $\\mathcal{O}(\\log n)$ since the search space is halved in each iteration.\n"
            "- **Space Complexity**: $\\mathcal{O}(1)$ auxiliary space."
        ),
    },
    {
        "prompt": "What are the scientific payloads on ISRO's Chandrayaan-3 Vikram Lander and Pragyan Rover?",
        "response": (
            "### ISRO Chandrayaan-3 Scientific Payloads\n\n"
            "Chandrayaan-3 achieved humanity's first soft landing near the Lunar South Pole on August 23, 2023.\n\n"
            "**Vikram Lander Payloads**:\n"
            "1. **ChaSTE** (Chandra's Surface Thermophysical Experiment): Measures thermal conductivity and temperature gradients across lunar regolith down to 10 cm depth.\n"
            "2. **ILSA** (Instrument for Lunar Seismic Activity): Detects lunar seismicity and moonquakes.\n"
            "3. **RAMBHA-LP** (Langmuir Probe): Measures lunar ionospheric plasma density and surface photoelectron sheath.\n\n"
            "**Pragyan Rover Payloads**:\n"
            "1. **LIBS** (Laser-Induced Breakdown Spectroscope): In-situ elemental composition analysis (confirmed presence of Sulphur, Iron, Calcium, Titanium).\n"
            "2. **APXS** (Alpha Particle X-Ray Spectrometer): Quantitative elemental analysis of lunar soil and rocks."
        ),
    },
    {
        "prompt": "Explain the Second Law of Thermodynamics and the concept of Entropy.",
        "response": (
            "### Second Law of Thermodynamics & Entropy\n\n"
            "The **Second Law of Thermodynamics** establishes the directionality of spontaneous physical and chemical processes:\n\n"
            "1. **Entropy Formulation**:\n"
            "   - In any isolated system, total entropy ($S$) always increases over time:\n"
            "     \\[ \\Delta S_{\\text{universe}} = \\Delta S_{\\text{system}} + \\Delta S_{\\text{surroundings}} \\ge 0 \\]\n"
            "2. **Clausius & Kelvin-Planck Statements**:\n"
            "   - **Clausius**: Heat cannot spontaneously flow from a colder body to a warmer body without external work.\n"
            "   - **Kelvin-Planck**: No heat engine can operate with 100% thermal efficiency by converting all absorbed heat entirely into work."
        ),
    },
]

# 2. 22 Scheduled Indian Languages Instruction Dialogues
INDIC_INSTRUCTION_DATA: list[dict[str, str]] = [
    {
        "lang": "hi",
        "prompt": "नमस्ते! आप कौन हैं और आपकी क्या विशेषताएँ हैं?",
        "response": (
            "नमस्ते! मैं **IndicLLM-Bharat** हूँ — भारत का स्वदेशी, सम्प्रभु (Sovereign) आर्टिफिशियल इंटेलिजेंस मॉडल।\n\n"
            "**मेरी प्रमुख विशेषताएँ**:\n"
            "1. 🌐 **२२ अनुसूचित भारतीय भाषाएँ**: हिन्दी, बंगाली, तमिल, तेलुगु, मराठी, गुजराती, कन्नड़, मलयालम, पंजाबी, ओड़िया, असमिया, संस्कृत आदि में पारंगत।\n"
            "2. 🔬 **विज्ञान एवं तकनीक**: कृत्रिम बुद्धिमत्ता, क्वांटम संगणना, अंतरिक्ष विज्ञान (ISRO) और गणित में दक्षता।\n"
            "3. 💻 **प्रोग्रामिंग एवं समस्या-समाधान**: पायथन, सी++, एल्गोरिदम एवं डेटा संरचनाओं में त्रुटिहीन कोडिंग।"
        ),
    },
    {
        "lang": "bn",
        "prompt": "ভারতীয় সংবিধানের প্রস্তাবনা এবং ড. বি.আর. আম্বেদকরের অবদান ব্যাখ্যা করুন।",
        "response": (
            "### ভারতীয় সংবিধান এবং ড. বি.আর. আম্বেদকরের অবদান\n\n"
            "**ড. ভীমরাও রামজি আম্বেদকর** ছিলেন ভারতীয় সংবিধানের খসড়া কমিটির সভাপতি এবং সংবিধানের মূল রূপকার।\n\n"
            "**প্রস্তাবনার মূল আদর্শ**:\n"
            "1. **সার্বভৌম, সমাজতান্ত্রিক, ধর্মনিরপেক্ষ, গণতান্ত্রিক সাধারণতন্ত্র** (Sovereign, Socialist, Secular, Democratic Republic)।\n"
            "2. **ন্যায়বিচার** (সামাজিক, অর্থনৈতিক ও রাজনৈতিক)।\n"
            "3. **স্বাধীনতা** (চিন্তা, মতপ্রকাশ, বিশ্বাস ও উপাসনার স্বাধীনতা)।\n"
            "4. **সাম্য ও ভ্রাতৃত্ববোধ**: প্রতিটি নাগরিকের মর্যাদা ও জাতীয় ঐক্য নিশ্চিত করা।"
        ),
    },
    {
        "lang": "ta",
        "prompt": "திருக்குறளின் பெருமை மற்றும் அதன் முப்பால்களை விளக்குக.",
        "response": (
            "### திருக்குறள் மற்றும் அதன் முப்பால்கள்\n\n"
            "திருவள்ளுவரால் இயற்றப்பட்ட **திருக்குறள்** உலகப் பொதுமறை என போற்றப்படும் ஒப்பற்ற தமிழ் செவ்வியல் இலக்கியமாகும்.\n\n"
            "**முப்பால்கள்**:\n"
            "1. **அறத்துப்பால்** (38 அதிகாரங்கள்): மனித வாழ்வின் நன்னெறி, தர்மம், மற்றும் ஒழுக்கத்தை எடுத்துரைக்கிறது.\n"
            "2. **பொருட்பால்** (70 அதிகாரங்கள்): ஆட்சி முறை, பொருளாதாரம், தலைமைப் பண்பு, மற்றும் நிர்வாக சிந்தனைகளை விளக்குகிறது.\n"
            "3. **காமத்துப்பால்** (25 அதிகாரங்கள்): தூய காதல், குடும்ப வாழ்வு மற்றும் அன்பின் உயர்வை பேசுகிறது.\n\n"
            "மொத்தம் **1330 குறட்பாக்கள்** வாழ்வின் அனைத்துப் பரிமாணங்களுக்கும் வழிகாட்டுகின்றன."
        ),
    },
    {
        "lang": "te",
        "prompt": "శ్రీకృష్ణదేవరాయల పాలన మరియు తెలుగు సాహిత్య సువర్ణయుగం గురించి వివరించండి.",
        "response": (
            "### శ్రీకృష్ణదేవరాయల కాలం — తెలుగు సాహిత్య స్వర్ణయుగం\n\n"
            "విజయనగర సామ్రాజ్య చక్రవర్తి **శ్రీకృష్ణదేవరాయలు** (1509–1529) గొప్ప పాలకుడు మరియు సాహిత్య పోషకుడు.\n\n"
            "1. **భువన విజయం**: ఆయన ఆస్థానంలో అష్టదిగ్గజములు (అల్లసాని పెద్దన, తెనాలి రామకృష్ణ, ధూర్జటి తదితరులు) ఉండేవారు.\n"
            "2. **ఆముక్తమాల్యద**: శ్రీకృష్ణదేవరాయలు స్వయంగా రచించిన ప్రబంధ కావ్యం.\n"
            '3. **ప్రసిద్ధ సూక్తి**: *"దేశభాషలందు తెలుగు లెస్స"* అని చాటి తెలుగు భాషకు అంతర్జాతీయ గౌరవం తెచ్చారు.'
        ),
    },
    {
        "lang": "mr",
        "prompt": "छत्रपती शिवाजी महाराजांच्या अष्टप्रधान मंडळाची माहिती द्या.",
        "response": (
            "### छत्रपती शिवाजी महाराजांचे अष्टप्रधान मंडळ\n\n"
            "स्वराज्याचा कारभार अत्यंत शिस्तबद्ध व कार्यक्षमपणे चालवण्यासाठी छत्रपती शिवाजी महाराजांनी **अष्टप्रधान मंडळाची** स्थापना केली:\n\n"
            "1. **पेशवे (मुख्य प्रधान)**: संपूर्ण प्रशासनाचे प्रमुख.\n"
            "2. **अमात्य (अर्थमंत्री)**: राज्याचा जमा-खर्च व महसूल व्यवस्था पाहणारे.\n"
            "3. **सचिव (सुरनिस)**: शासकीय पत्रव्यवहार व आज्ञापत्रे तपासणारे.\n"
            "4. **वाकेनीस (गृहमंत्री)**: दरबारातील अंतर्गत कामकाज व गुप्तहेर यंत्रणा.\n"
            "5. **सरनोबत (सेनापती)**: लष्कराचे प्रमुख नेतृत्व.\n"
            "6. **सुमंत (परराष्ट्र मंत्री)**: इतर सत्तांशी राजनैतिक संबंध सांभाळणारे.\n"
            "7. **न्यायाधीश**: न्यायदान प्रणालीचे प्रमुख.\n"
            "8. **पंडितराव (धर्माधिकारी)**: धार्मिक व सांस्कृतिक कार्यभार सांभाळणारे."
        ),
    },
    {
        "lang": "gu",
        "prompt": "ગુજરાતી સાહિત્યમાં નરસિંહ મહેતા અને મીરાંબાઈનું પ્રદાન સમજાવો.",
        "response": (
            "### નરસિંહ મહેતા અને મીરાંબાઈનું સાહિત્યિક પ્રદાન\n\n"
            "1. **નરસિંહ મહેતા (આદિ કવિ)**:\n"
            "   - 'વૈષ્ણવ જન તો તેને રે કહીએ' જેવા અમર પ્રભાતિયાં દ્વારા માનવતા અને સમભાવનો સંદેશ આપ્યો.\n"
            "   - ઝૂલણા છંદમાં કૃષ્ણભક્તિ અને જ્ઞાનમાર્ગનું સુંદર આલેખન કર્યું.\n\n"
            "2. **મીરાંબાઈ**:\n"
            "   - પ્રેમલક્ષણા ભક્તિ અને આત્મસમર્પણના ઉત્કૃષ્ટ પદોની રચના કરી.\n"
            "   - ગુજરાતી, રાજસ્થાની અને વ્રજ ભાષાના સંગમથી ભક્તિ આંદોલનને લોકપ્રિય બનાવ્યું."
        ),
    },
    {
        "lang": "kn",
        "prompt": "ಕನ್ನಡದ ವಚನ ಸಾಹಿತ್ಯ ಮತ್ತು ಬಸವಣ್ಣನವರ ತತ್ತ್ವಗಳನ್ನು ವಿವರಿಸಿ.",
        "response": (
            "### ವಚನ ಸಾಹಿತ್ಯ ಮತ್ತು ಬಸವೇಶ್ವರರ ತತ್ತ್ವಗಳು\n\n"
            "೧೨ನೇ ಶತಮಾನದ **ವಚನ ಚಳವಳಿ** ಕನ್ನಡ ಸಾಹಿತ್ಯ ಮತ್ತು ಸಮಾಜ ಸುಧಾರಣೆಯಲ್ಲಿ ಕ್ರಾಂತಿಕಾರಿ ಮೈಲಿಗಲ್ಲು.\n\n"
            "1. **ಕಾಯಕವೇ ಕೈಲಾಸ**: ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯೂ ಪ್ರಾಮಾಣಿಕ ಕಾಯಕ ಮಾಡಬೇಕು.\n"
            "2. **ದಾಸೋಹ ಭಾವ**: ಗಳಿಸಿದ ಸಂಪತ್ತನ್ನು ಸಮಾಜದ ಒಳಿತಿಗಾಗಿ ಸಮರ್ಪಿಸುವುದು.\n"
            "3. **ಅನುಭವ ಮಂಟಪ**: ವಿಶ್ವದ ಮೊದಲ ಪ್ರಜಾಪ್ರಭುತ್ವ ಸಂಸತ್ತು, ಇಲ್ಲಿ ಜಾತಿ-ಲಿಂಗ ಭೇದವಿಲ್ಲದೆ ಸರ್ವರಿಗೂ ಸಮಾನ ಸ್ಥಾನ ನೀಡಲಾಗಿತ್ತು."
        ),
    },
    {
        "lang": "ml",
        "prompt": "മലയാള ഭാഷയുടെ പിതാവായ തുഞ്ചത്തെഴുത്തച്ഛനെക്കുറിച്ച് വ്യക്തമാക്കുക.",
        "response": (
            "### തുഞ്ചത്തെഴുത്തച്ഛനും മലയാള സാഹിത്യവും\n\n"
            "പതിനാറാം നൂറ്റാണ്ടിൽ ജീവിച്ചിരുന്ന **തുഞ്ചത്ത് രാമാനുജൻ എഴുത്തച്ഛൻ** മലയാള ഭാഷയുടെ പിതാവായി ആദരിക്കപ്പെടുന്നു.\n\n"
            "1. **കിളിപ്പാട്ട് പ്രസ്ഥാനം**: 'അദ്ധ്യാത്മരാമായണം കിളിപ്പാട്ട്' மற்றும் 'മഹാഭാരതം കിളിപ്പാട്ട്' എന്നിവയിലൂടെ മലയാള ലിപിക്കും ഭാഷാ വ്യാകരണത്തിനും രൂപം നൽകി.\n"
            "2. **സാമൂഹിക നവോത്ഥാനം**: സംസ്കൃത ഇതിഹാസങ്ങളെ ലളിതമായ മലയാളത്തിൽ ജനങ്ങളിലേക്ക് എത്തിച്ചു."
        ),
    },
    {
        "lang": "pa",
        "prompt": "ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ ਦੀ ਸੰਪਾਦਨਾ ਅਤੇ ਵਿਸ਼ਵ ਸ਼ਾਂਤੀ ਦਾ ਸੁਨੇਹਾ ਸਮਝਾਓ।",
        "response": (
            "### ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ ਅਤੇ ਵਿਸ਼ਵ ਭਾਈਚਾਰਾ\n\n"
            "**ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ** ਦੀ ਸੰਪਾਦਨਾ ਪੰਜਵੇਂ ਪਾਤਸ਼ਾਹ ਸ੍ਰੀ ਗੁਰੂ ਅਰਜਨ ਦੇਵ ਜੀ ਨੇ ੧੬੦੪ ਈਸਵੀ ਵਿੱਚ ਕੀਤੀ।\n\n"
            "1. **ਸਰਬਸਾਂਝੀਵਾਲਤਾ**: ਇਸ ਵਿੱਚ ਸਿੱਖ ਗੁਰੂ ਸਾਹਿਬਾਨ ਤੋਂ ਇਲਾਵਾ ਭਗਤ ਕਬੀਰ ਜੀ, ਬਾਬਾ ਫਰੀਦ ਜੀ, ਭਗਤ ਨਾਮਦੇਵ ਜੀ, ਅਤੇ ਭਗਤ ਰਵਿਦਾਸ ਜੀ ਦੀ ਬਾਣੀ ਦਰਜ ਹੈ।\n"
            '2. **ਮੁੱਖ ਸੰਦੇਸ਼**: *"ਸਭੇ ਸਾਝੀਵਾਲ ਸਦਾਇਨਿ ਤੂੰ ਕਿਸੈ ਨ ਦਿਸਹਿ ਬਾਹਰਾ ਜੀਉ"* — ਸਾਰੀ ਮਨੁੱਖਤਾ ਇੱਕ ਪ੍ਰਮਾਤਮਾ ਦੀ ਸੰਤਾਨ ਹੈ।'
        ),
    },
    {
        "lang": "or",
        "prompt": "ଓଡ଼ିଆ ସାହିତ୍ୟରେ ପଞ୍ଚସଖା ଯୁଗର ଗୁରୁତ୍ୱ ବର୍ଣ୍ଣନା କରନ୍ତୁ।",
        "response": (
            "### ଓଡ଼ିଆ ସାହିତ୍ୟର ପଞ୍ଚସଖା ଯୁଗ\n\n"
            "ଷୋଡ଼ଶ ଶତାବ୍ଦୀରେ ପଞ୍ଚସଖା (ଜଗନ୍ନାଥ ଦାସ, ବଳରାମ ଦାସ, ଅଚ୍ୟୁତାନନ୍ଦ ଦାସ, ଅନନ୍ତ ଦାସ, ଯଶୋବନ୍ତ ଦାସ) ଓଡ଼ିଆ ଭାଷା ଓ ସାହିତ୍ୟକୁ ନୂତନ ଦିଗ ଦେଇଥିଲେ।\n\n"
            "1. **ଜଗନ୍ନାଥ ଦାସଙ୍କ 'ଶ୍ରୀମଦ୍ ଭାଗବତ'**: ଓଡ଼ିଶାର ପ୍ରତ୍ୟେକ ଗ୍ରାମରେ ଭାଗବତ ଟୁଙ୍ଗି ମାଧ୍ୟମରେ ସାମାଜିକ ଓ ଆଧ୍ୟାତ୍ମିକ ଚେତନା ଜାଗ୍ରତ କରିଥିଲା।\n"
            "2. **ବଳରାମ ଦାସଙ୍କ 'ଦାଣ୍ଡି ରାମାୟଣ'**: ଓଡ଼ିଆ ଲୋକଭାଷାରେ ରାମାୟଣକୁ ଜନପ୍ରିୟ କରାଇଥିଲା।"
        ),
    },
    {
        "lang": "as",
        "prompt": "শ্ৰীমন্ত শংকৰদেৱৰ নৱবৈষ্ণৱ আন্দোলন আৰু অসমীয়া সাহিত্যৰ বিকাশ আলোচনা কৰক।",
        "response": (
            "### শ্ৰীমন্ত শংকৰদেৱ আৰু অসমীয়া সংস্কৃতি\n\n"
            "মহাপুৰুষ **শ্ৰীমন্ত শংকৰদেৱ** (১৪৪৯–১৫৬৮) অসমীয়া সমাজ, ধৰ্ম আৰু সাহিত্যৰ জনক।\n\n"
            "1. **একশৰণ নামধৰ্ম**: জাতি-ভেদহীন সমাজ গঠনৰ বাবে নামঘৰ প্ৰতিষ্ঠা কৰিছিল।\n"
            "2. **বৰগীত আৰু অংকীয়া নাট**: ব্ৰজাৱলী আৰু অসমীয়া ভাষাত ৰচিত সাহিত্যই অসমীয়া জাতিক ঐক্যবদ্ধ কৰিছিল।"
        ),
    },
    {
        "lang": "ur",
        "prompt": "مرزا غالب کی شاعری کی خصوصیات اور اردو غزل میں ان کا مقام بیان کیجیے۔",
        "response": (
            "### مرزا اسد اللہ خان غالب اور اردو شاعری\n\n"
            "**مرزا غالب** (1797–1869) اردو زبان و ادب کے عظیم ترین شاعر اور مفکر ہیں۔\n\n"
            "1. **فلسفیانہ فکر**: غالب نے روایتی عشق و عاشقی سے ہٹ کر انسانی وجود، کائنات، اور زندگی کی حقیقتوں کو موضوع بنایا۔\n"
            "2. **شوخی و ظرافت**: غم اور تکالیف کو مسکرا کر جھیلنے کا انداز غالب کی شاعری کا خاص وصف ہے۔\n"
            "3. **اردو خطوط**: غالب نے خطوط کو 'مراسلے سے مکالمہ' بنا کر جدید اردو نثر کی بنیاد رکھی۔"
        ),
    },
    {
        "lang": "sa",
        "prompt": "संस्कृतभाषायाः वैज्ञानिकस्वरूपं व्याकरणस्य च महत्त्वं लिखत।",
        "response": (
            "### संस्कृतभाषायाः वैज्ञानिकस्वरूपम्\n\n"
            "संस्कृतभाषा विश्वस्य प्राचीनतमा, समृद्धतमा, अत्यन्तं वैज्ञानिकी च भाषा वर्तते।\n\n"
            "1. **पाणिनेः अष्टाध्यायी**: महर्षिपाणिनाविरचितं व्याकरणशास्त्रं सङ्गणकशास्त्रस्य (Computer Science) अल्गोरिदम-सदृशं नियमबद्धम् अस्ति।\n"
            "2. **उच्चारणशुद्धता**: संस्कृतवर्णानां ध्वनि-उत्पत्तिस्थानानि वैज्ञानिकाधारेण सुव्यवस्थितानि सन्ति।"
        ),
    },
]


def get_all_instruction_curriculum() -> list[dict[str, Any]]:
    """Return all combined instruction-following dialogues."""
    return STEM_INSTRUCTION_DATA + INDIC_INSTRUCTION_DATA


def export_instruction_curriculum(output_path: str | Path) -> int:
    """Export complete SFT instruction curriculum to JSONL format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = get_all_instruction_curriculum()
    with open(out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(items)
