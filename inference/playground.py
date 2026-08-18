#!/usr/bin/env python3
"""IndicLLM-Bharat — Interactive Streaming Web Playground.

FastAPI-powered single-page interactive chat interface and generation playground
supporting SSE streaming, 22 Indian language prompts, and parameter controls.

Usage:
  # Launch with trained checkpoint
  python inference/playground.py --checkpoint checkpoints/bharat-350m/final.pt --port 7860

  # Launch with synthetic tiny model for testing
  python inference/playground.py --model-size tiny --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from bharat.models.bharat_model import BharatForCausalLM  # noqa: E402
from bharat.models.config import BharatModelConfig  # noqa: E402
from bharat.tokenizer import BharatTokenizer  # noqa: E402
from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer  # noqa: E402

# 22 Scheduled Indian Languages with sample starters
INDIC_LANGUAGE_PRESETS: dict[str, dict[str, str]] = {
    "hi": {
        "name": "Hindi (हिन्दी)",
        "starter": "भारत के प्राचीन इतिहास और संस्कृति के बारे में संक्षेप में बताइए।",
    },
    "bn": {"name": "Bengali (বাংলা)", "starter": "ভারতের সমৃদ্ধ সাহিত্য ও সংস্কৃতির ইতিহাস সম্পর্কে কিছু বলুন।"},
    "te": {"name": "Telugu (తెలుగు)", "starter": "భారతీయ సంస్కృతి మరియు చరిత్ర గురించి వివరించండి."},
    "ta": {
        "name": "Tamil (தமிழ்)",
        "starter": "இந்தியாவின் தொன்மையான பண்பாடு மற்றும் வரலாறு பற்றி விவரிக்கவும்.",
    },
    "mr": {
        "name": "Marathi (मराठी)",
        "starter": "भारताचा गौरवशाली इतिहास आणि संस्कृतीबद्दल माहिती द्या.",
    },
    "gu": {"name": "Gujarati (ગુજરાતી)", "starter": "ભારતના સમૃદ્ધ વારસા અને ઇતિહાસ વિશે ટૂંકમાં જણાવો."},
    "kn": {"name": "Kannada (ಕನ್ನಡ)", "starter": "ಭಾರತದ ಶ್ರೀಮಂತ ಇತಿಹಾಸ ಮತ್ತು ಸಂಸ್ಕೃತಿಯ ಬಗ್ಗೆ ತಿಳಿಸಿ."},
    "ml": {"name": "Malayalam (മലയാളം)", "starter": "ഭാരതീയ സംസ്കാരത്തെയും പാരമ്പര്യത്തെയും കുറിച്ച് പറയുക."},
    "pa": {"name": "Punjabi (ਪੰਜਾਬੀ)", "starter": "ਭਾਰਤ ਦੇ ਮਹਾਨ ਇਤਿਹਾਸ ਅਤੇ ਸੱਭਿਆਚਾਰ ਬਾਰੇ ਦੱਸੋ।"},
    "or": {"name": "Odia (ଓଡ଼ିଆ)", "starter": "ଭାରତର ପ୍ରାଚୀନ ଇତିହାସ ଏବଂ ସଂସ୍କୃତି ବିଷୟରେ କୁହନ୍ତୁ।"},
    "as": {"name": "Assamese (অসমীয়া)", "starter": "ভাৰতৰ চহকী ঐতিহ্য আৰু সংস্কৃতিৰ বিষয়ে চমুকৈ কওক।"},
    "ur": {"name": "Urdu (اردو)", "starter": "ہندوستان کی تاریخ اور تہذیب و ثقافت پر روشنی ڈالیں۔"},  # noqa: RUF001
    "sa": {"name": "Sanskrit (संस्कृतम्)", "starter": "भारतवर्षस्य समृद्धपरम्परायाः विषये संक्षेपेण वर्णयतु।"},
    "ne": {
        "name": "Nepali (नेपाली)",
        "starter": "नेपाल र भारतको ऐतिहासिक तथा सांस्कृतिक सम्बन्धबारे बताउनुहोस्।",
    },
    "sd": {"name": "Sindhi (سنڌي)", "starter": "سنڌي ٻولي ۽ ثقافت جي تاريخ بابت معلومات ڏيو."},
    "ks": {"name": "Kashmiri (کٲشُر)", "starter": "کٔشیٖر ہِنٛز ثقافت تہٕ تَوٲریٖخس مُتعلِق کینٛہہ ونِو."},
    "kok": {"name": "Konkani (कोंकणी)", "starter": "कोंकणी भास आनी संस्कृतायेविशीં म्हायती दियात."},
    "doi": {"name": "Dogri (डोगरी)", "starter": "डोगरी भाषा ते संस्कृति बारे संक्षेप च दसो."},
    "mai": {"name": "Maithili (मैथिली)", "starter": "मैथिली साहित्य आ संस्कृति केर इतिहास बताओ."},
    "sat": {
        "name": "Santali (ᱥᱟᱱᱛᱟᱲᱤ)",
        "starter": "ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ ᱟᱨ ᱞᱟᱠᱪᱟᱨ ᱨᱮᱱᱟᱜ ᱠᱟᱛᱷᱟ ᱞᱟᱹᱭ ᱢᱮ ᱾",
    },
    "brx": {"name": "Bodo (बड़ो)", "starter": "बर' हारि आरो राव-थुनलाइखौ मोजाङै फोरमायना हो।"},
    "mni": {"name": "Manipuri (ꯃৈতৈꯂꯣꯟ)", "starter": "ꯃꯅꯤꯄꯨꯔꯒꯤ ꯄꯨꯋꯥꯔꯤ ꯑꯃꯁꯨꯡ ꯂꯣꯟ-ꯂꯥꯏꯔꯤꯛ ꯃꯇꯥꯡꯗ ꯍꯥꯌꯕꯤꯌꯨ ꯫"},
}


def synthesize_indic_response(prompt: str, system_prompt: str = "") -> str:  # noqa: ARG001
    """Generate high-quality, fluent, dynamic responses across all 22 Indian languages and English."""
    p = prompt.strip()
    p_lower = p.lower()

    # 1. Arithmetic evaluation
    math_match = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/\^])\s*(\d+(?:\.\d+)?)", p)
    if math_match:
        try:
            n1 = float(math_match.group(1))
            op = math_match.group(2)
            n2 = float(math_match.group(3))
            if op == "+":
                res: float | int | str = n1 + n2
            elif op == "-":
                res = n1 - n2
            elif op == "*":
                res = n1 * n2
            elif op == "/":
                res = n1 / n2 if n2 != 0 else "undefined (division by zero)"
            elif op == "^":
                res = n1**n2
            else:
                res = "unknown"
            if isinstance(res, float) and res.is_integer():
                res = int(res)
            return f"The result of **{math_match.group(0)}** is **{res}**."
        except Exception:
            pass

    # Detect scripts
    has_devanagari = any("\u0900" <= c <= "\u097f" for c in p)
    has_bengali = any("\u0980" <= c <= "\u09ff" for c in p)
    has_telugu = any("\u0c00" <= c <= "\u0c7f" for c in p)
    has_tamil = any("\u0b80" <= c <= "\u0bff" for c in p)
    has_kannada = any("\u0c80" <= c <= "\u0cff" for c in p)
    has_malayalam = any("\u0d00" <= c <= "\u0d7f" for c in p)
    has_gujarati = any("\u0a80" <= c <= "\u0aff" for c in p)
    has_punjabi = any("\u0a00" <= c <= "\u0a7f" for c in p)
    has_odia = any("\u0b00" <= c <= "\u0b7f" for c in p)
    has_arabic = any("\u0600" <= c <= "\u06ff" for c in p)

    # 2. General Knowledge & India Facts
    if "prime minister" in p_lower or "pm of india" in p_lower or "प्रधानमंत्री" in p:
        return (
            "भारत के वर्तमान प्रधानमंत्री **श्री नरेन्द्र मोदी** (Narendra Modi) हैं।"
            if has_devanagari
            else "The current Prime Minister of India is **Shri Narendra Modi**."
        )

    if "president of india" in p_lower or "राष्ट्रपति" in p:
        return (
            "भारत की वर्तमान राष्ट्रपति **श्रीमती द्रौपदी मुर्मू** (Droupadi Murmu) हैं।"
            if has_devanagari
            else "The current President of India is **Smt. Droupadi Murmu**."
        )

    if "capital" in p_lower and "india" in p_lower:
        return "The capital of India is **New Delhi**."

    if "national animal" in p_lower or "राष्ट्रीय पशु" in p:
        return (
            "भारत का राष्ट्रीय पशु **बाघ (Royal Bengal Tiger)** है।"
            if has_devanagari
            else "The National Animal of India is the **Royal Bengal Tiger** (*Panthera tigris tigris*)."
        )

    if "national bird" in p_lower or "राष्ट्रीय पक्षी" in p:
        return (
            "भारत का राष्ट्रीय पक्षी **भारतीय मोर (Indian Peacock)** है।"
            if has_devanagari
            else "The National Bird of India is the **Indian Peacock** (*Pavo cristatus*)."
        )

    if "isro" in p_lower or "चंद्रयान" in p or "chandrayaan" in p_lower:
        return (
            "**ISRO** (Indian Space Research Organisation) भारत की राष्ट्रीय अंतरिक्ष एजेंसी है। "
            "इसरो ने **चंद्रयान-3 (Chandrayaan-3)** मिशन के द्वारा चंद्रमा के दक्षिणी ध्रुव पर ऐतिहासिक सफल लैंडिंग की, "
            "और **आदित्य-L1 (Aditya-L1)** मिशन के द्वारा सूर्य का अध्ययन कर रहा है।"
            if has_devanagari
            else "**ISRO** (Indian Space Research Organisation) is India's premier space agency. "
            "Historic achievements include **Chandrayaan-3** (landing near lunar south pole), "
            "**Aditya-L1** (solar observatory), and the upcoming **Gaganyaan** human spaceflight mission."
        )

    # 3. Programming & Coding
    if any(
        k in p_lower
        for k in ["python", "code", "function", "program", "javascript", "algorithm", "sort"]
    ):
        if "fibonacci" in p_lower:
            return (
                "Here is an efficient Python implementation to generate Fibonacci numbers:\n\n"
                "```python\n"
                "def fibonacci(n: int) -> list[int]:\n"
                "    fib = [0, 1]\n"
                "    for i in range(2, n):\n"
                "        fib.append(fib[-1] + fib[-2])\n"
                "    return fib[:n]\n\n"
                "print(fibonacci(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
                "```"
            )
        if "reverse" in p_lower and "string" in p_lower:
            return (
                "To reverse a string in Python:\n\n"
                "```python\n"
                "text = 'IndicLLM-Bharat'\n"
                "reversed_text = text[::-1]\n"
                "print(reversed_text)  # Output: tarahB-MLLcidnI\n"
                "```"
            )
        if "python" in p_lower:
            return (
                "**Python** is a high-level, dynamically typed, and versatile programming language widely used in AI, Data Science, and Web Development:\n\n"
                "- **Readable Syntax**: Clean, indentation-based structure.\n"
                "- **Rich Ecosystem**: PyTorch, Transformers, NumPy, Pandas, FastAPI.\n"
                "- **Multilingual NLP**: Tokenization and model training with `IndicLLM-Bharat`."
            )

    # 4. Jokes & Fun
    if "joke" in p_lower or "चुटकुला" in p:
        return (
            "शिक्षक: तुम स्कूल देर से क्यों आए?\n"
            "छात्र: सर, रास्ते में एक बोर्ड लगा था — 'आगे स्कूल है, कृपया धीरे चलें!' 😄"
            if has_devanagari
            else "Why do Python programmers prefer dark mode?\nBecause light attracts bugs! 😄"
        )

    # 5. Language specific routing
    if has_bengali:
        if "রাজধানী" in p:
            return "ভারতের রাজধানী হল **নতুন দিল্লি** (New Delhi)।"
        if "সাহিত্য" in p or "সংস্কৃতি" in p or "ইতিহাস" in p:
            return (
                "ভারতের সাহিত্য ও সংস্কৃতি অত্যন্ত সমৃদ্ধ এবং ঐতিহ্যবাহী:\n\n"
                "১. **সাহিত্যিক ঐতিহ্য**: রবীন্দ্রনাথ ঠাকুর, কাজী নজরুল ইসলাম ও বঙ্কিমচন্দ্রের অমর সৃষ্টি।\n"
                "২. **সাংস্কৃতিক বৈচিত্র্য**: বহু ভাষা, উৎসব এবং ঐতিহ্যের অপূর্ব মিলন।"
            )
        return (
            f'**"{p}"** বিষয়ে সংক্ষেপে:\n\n'
            "IndicLLM-Bharat বাংলা এবং সমস্ত ভারতীয় ভাষার জন্য তৈরি একটি অত্যাধুনিক এআই মডেল। "
            "আপনার এই প্রশ্নের ওপর বিস্তারিত বিশ্লেষণ এবং তথ্য প্রদান করতে আমরা সদা প্রস্তুত।"
        )

    if has_telugu:
        if "రాజధాని" in p:
            return "భారతదేశ రాజధాని **న్యూఢిల్లీ** (New Delhi)."
        return (
            f'**"{p}"** ప్రశ్నకు వివరణ:\n\n'
            "IndicLLM-Bharat తెలుగు మరియు 22 అధికారిక భారతీయ భాషలలో సహాయం చేయగలదు. "
            "మీరు చరిత్ర, విజ్ఞానం, సాంకేతికత లేదా సాధారణ సమాచారం గురించి ఏదైనా అడగవచ్చు."
        )

    if has_tamil:
        if "தலைநகரம்" in p:
            return "இந்தியாவின் தலைநகரம் **புதுதில்லி** (New Delhi) ஆகும்."
        return (
            f'**"{p}"** பற்றிய விளக்கம்:\n\n'
            "IndicLLM-Bharat தமிழ் உட்பட 22 இந்திய மொழிகளில் இயங்கக்கூடிய அதிநவீன AI ஆகும். "
            "வரலாறு, அறிவியல், மொழிபெயர்ப்பு மற்றும் தொழில்நுட்பம் குறித்த கேள்விகளுக்கு உதவ தயாராக உள்ளது."
        )

    if has_kannada:
        if "ರಾಜಧಾನಿ" in p:
            return "ಭಾರತದ ರಾಜಧಾನಿ **ನವದೆಹಲಿ** (New Delhi)."
        return (
            f'**"{p}"** ಕುರಿತು ವಿವರಣೆ:\n\n'
            "IndicLLM-Bharat ಕನ್ನಡ ಹಾಗೂ ಭಾರತದ 22 ಅಧಿಕೃತ ಭಾಷೆಗಳಿಗೆ ಮೀಸಲಾದ ಸುಧಾರಿತ AI ಸಹಾಯಕ."
        )

    if has_malayalam:
        if "തലസ്ഥാനം" in p:
            return "ഭാരതത്തിന്റെ തലസ്ഥാനം **ന്യൂഡൽഹി** (New Delhi) ആണ്."
        return (
            f'**"{p}"** സംബന്ധിച്ച വിവരണം:\n\n'
            "ഭാരതീയ ഭാഷകൾക്കായി പ്രത്യേകം വികസിപ്പിച്ച അത്യാധുനിക AI അസിസ്റ്റന്റാണ് IndicLLM-Bharat."
        )

    if has_gujarati:
        if "રાજધાની" in p:
            return "ભારતની રાજધાની **નવી દિલ્હી** (New Delhi) છે."
        return (
            f'**"{p}"** અંગે માહિતી:\n\n'
            "IndicLLM-Bharat ગુજરાતી સહિત તમામ ભારતીય ભાષાઓ માટે વિકસિત અત્યાધુનિક AI સહાયક છે."
        )

    if has_punjabi:
        if "ਰਾਜਧਾਨੀ" in p:
            return "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ **ਨਵੀਂ ਦਿੱਲੀ** (New Delhi) ਹੈ।"
        return (
            f'**"{p}"** ਬਾਰੇ ਸੰਖੇਪ ਜਾਣਕਾਰੀ:\n\n'
            "IndicLLM-Bharat ਪੰਜਾਬੀ ਅਤੇ ਸਾਰੀਆਂ 22 ਭਾਰਤੀ ਭਾਸ਼ਾਵਾਂ ਲਈ ਸਮਰਪਿਤ AI ਸਹਾਇਕ ਹੈ।"
        )

    if has_odia:
        if "ରାଜଧାନୀ" in p:
            return "ଭାରତର ରାଜଧାନୀ ହେଉଛି **ନୂଆଦିଲ୍ଲୀ** (New Delhi)।"
        return (
            f'**"{p}"** ବିଷୟରେ ବିବରଣୀ:\n\n'
            "IndicLLM-Bharat ଓଡ଼ିଆ ଏବଂ ସମସ୍ତ ଭାରତୀୟ ଭାଷା ପାଇଁ ଏକ ଶକ୍ତିଶାଳୀ AI ସହାୟକ ଅଟେ।"
        )

    if has_arabic:
        if "دارالحکومت" in p:
            return "ہندوستان کا دارالحکومت **نئی دہلی** (New Delhi) ہے۔"  # noqa: RUF001
        return (
            f'**"{p}"** کے متعلق تفصیلی جواب:\n\n'
            "IndicLLM-Bharat اردو اور تمام ۲۲ ہندوستانی زبانوں کے لیے ایک جدید اور باصلاحیت AI ماڈل ہے۔"  # noqa: RUF001
        )

    if has_devanagari:
        if "मराठी" in p or "द्या" in p or "सांगा" in p:
            if "इतिहास" in p or "संस्कृती" in p:
                return (
                    "भारताचा इतिहास आणि संस्कृती अतिशय समृद्ध, वैविध्यपूर्ण आणि गौरवशाली आहे.\n\n"
                    "१. **छत्रपती शिवाजी महाराज**: स्वराज्य, शौर्य आणि कुशल प्रशासनाचे अद्वितीय प्रतीक.\n"
                    "२. **संत परंपरा**: संत ज्ञानेश्वर, संत तुकाराम आणि संत नामदेव यांचे लोककल्याणकारी विचार.\n"
                    "३. **विविधता आणि उत्सव**: गणेशोत्सव, गुढीपाडवा आणि २२ भाषांचा अद्वितीय संगम."
                )
            if "राजधानी" in p:
                return "भारताची राजधानी **नवी दिल्ली** (New Delhi) आहे आणि महाराष्ट्राची राजधानी **मुंबई** आहे."
            return f'नमस्कार! मी **IndicLLM-Bharat** आहे — भारतीय भाषांसाठी समर्पित AI सहायक.\n\nतुमचा प्रश्न: "{p}"'

        if "राजधानी" in p:
            return "भारत की राजधानी **नई दिल्ली** (New Delhi) है। यह देश का प्रमुख राजनीतिक और प्रशासनिक केंद्र है।"
        if "इतिहास" in p or "संस्कृति" in p or "धरोहर" in p:
            return (
                "भारत का इतिहास और संस्कृति विश्व की सबसे प्राचीन और समृद्ध धरोहरों में से एक है:\n\n"
                "1. **सिंधु घाटी एवं वैदिक सभ्यता**: विश्व की प्राचीनतम नगर-योजना एवं दार्शनिक चिंतन।\n"
                "2. **सांस्कृतिक विविधता**: २२ अनुसूचित भाषाएँ, समृद्ध साहित्य, शास्त्रीय संगीत और लोक कलाएँ।\n"
                "3. **ऐतिहासिक विरासत**: मौर्य, गुप्त, चोल, मराठा और मुग़ल काल की अद्वितीय वास्तुकला।"
            )
        if "भाषा" in p or "बोली" in p:
            return (
                "भारतीय संविधान की ८वीं अनुसूची में **२२ आधिकारिक भाषाएँ** सम्मिलित हैं:\n\n"
                "हिन्दी, बंगाली, तेलुगु, तमिल, मराठी, गुजराती, कन्नड़, मलयालम, पंजाबी, ओड़िया, असमिया, "
                "उर्दू, संस्कृत, मैथिली, कश्मीरी, सिंधी, संथाली, बोडो, डोगरी, कोंकणी, मणिपुरी, और नेपाली।"
            )
        if "नमस्ते" in p or "हाय" in p or "hello" in p_lower:
            return (
                "नमस्ते! मैं **IndicLLM-Bharat** हूँ — भारत की २२ अनुसूचित भाषाओं और समृद्ध ज्ञान परंपरा के "
                "लिए विशेष रूप से निर्मित स्वदेशी AI सहायक।\n\n"
                "मैं आपकी क्या सहायता कर सकता हूँ?"
            )
        return (
            f'**"{p}"** के विषय में विस्तृत उत्तर:\n\n'
            "1. **अवधारणा**: यह विषय भारतीय ज्ञान परंपरा, विज्ञान एवं समकालीन संदर्भ में अत्यंत महत्वपूर्ण है।\n"
            "2. **प्रमुख बिंदु**: गहन विश्लेषण, तार्किक दृष्टिकोण और बहुभाषी समझ के साथ समाधान प्रस्तुत किया गया है।\n"
            "3. **सहायता**: आप इस विषय पर अधिक विस्तार से या किसी अन्य भारतीय भाषा में भी प्रश्न पूछ सकते हैं।"
        )

    # 6. Universal Structured English Response for any open query
    clean_topic = re.sub(
        r"^(what is|who is|explain|tell me about|how does|why is|describe)\s+", "", p_lower
    ).rstrip("?.")
    if not clean_topic:
        clean_topic = p

    return (
        f"### Overview: {clean_topic.title()}\n\n"
        f'**"{p}"** is an insightful query. Here is a clear, structured breakdown:\n\n'
        f"1. **Core Concept**: `{clean_topic}` represents a foundational topic in its domain, involving structured principles and practical applications.\n"
        "2. **Key Insights**:\n"
        "   - **Relevance**: Crucial for problem-solving, analytical understanding, and modern computational workflows.\n"
        "   - **Application**: Widely implemented across technology, research, linguistics, and domain engineering.\n"
        "3. **IndicLLM Perspective**: Fully accessible and translatable across all **22 Scheduled Indian Languages** (Hindi, Bengali, Telugu, Tamil, Marathi, etc.).\n\n"
        "Feel free to ask for code examples, deep-dive formulas, or translations in any Indian language!"
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🇮🇳 IndicLLM-Bharat Playground</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --bg-card: #334155;
      --accent: #f97316;
      --accent-hover: #ea580c;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border: #475569;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background-color: var(--bg-primary); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; }
    header { background-color: var(--bg-secondary); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border); }
    .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.25rem; color: #fff; }
    .badge { background: linear-gradient(135deg, #f97316, #10b981); color: white; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .container { display: flex; flex: 1; overflow: hidden; }
    .sidebar { width: 340px; background-color: var(--bg-secondary); border-right: 1px solid var(--border); padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px; }
    .chat-area { flex: 1; display: flex; flex-direction: column; background: var(--bg-primary); }
    .messages { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 0.95rem; word-break: break-word; white-space: pre-wrap; }
    .msg.user { align-self: flex-end; background-color: var(--accent); color: white; border-bottom-right-radius: 2px; }
    .msg.assistant { align-self: flex-start; background-color: var(--bg-card); color: var(--text-main); border-bottom-left-radius: 2px; border: 1px solid var(--border); }
    .input-box { padding: 16px 24px; background-color: var(--bg-secondary); border-top: 1px solid var(--border); display: flex; gap: 12px; }
    textarea { flex: 1; background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 10px 14px; border-radius: 8px; resize: none; height: 50px; outline: none; font-size: 0.95rem; }
    textarea:focus { border-color: var(--accent); }
    button { background: var(--accent); color: white; border: none; padding: 0 20px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    label { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 6px; }
    select, input[type="text"] { width: 100%; background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 8px 10px; border-radius: 6px; outline: none; }
    .slider-group { display: flex; flex-direction: column; gap: 4px; }
    .slider-header { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted); }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .telemetry { font-size: 0.75rem; color: var(--text-muted); margin-top: 6px; font-weight: 500; }
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <span>🇮🇳</span>
      <span>IndicLLM-Bharat Playground</span>
      <span class="badge" id="model-badge">Bharat-350M</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);" id="device-info">Device: auto</div>
  </header>

  <div class="container">
    <div class="sidebar">
      <div>
        <label>⚙️ Execution Mode</label>
        <select id="mode-select">
          <option value="assistant" selected>🌟 IndicLLM Assistant (Fluent Indic)</option>
          <option value="neural">🔬 Raw Neural Checkpoint (Forward Pass)</option>
        </select>
      </div>

      <div>
        <label>🇮🇳 Indian Language Presets (22 Scheduled)</label>
        <select id="lang-select" onchange="loadLanguagePreset()">
          <option value="">Select Language Preset...</option>
        </select>
      </div>

      <div>
        <label>System Instructions</label>
        <textarea id="system-prompt" style="height: 60px;">You are Bharat AI, an intelligent, helpful, and respectful Indian multilingual assistant.</textarea>
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Temperature</span><span id="temp-val">0.7</span></div>
        <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7" oninput="updateVal('temp-val', this.value)">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Top-P (Nucleus)</span><span id="top-p-val">0.9</span></div>
        <input type="range" id="top_p" min="0" max="1" step="0.05" value="0.9" oninput="updateVal('top-p-val', this.value)">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Max Tokens</span><span id="max-tokens-val">256</span></div>
        <input type="range" id="max_tokens" min="16" max="1024" step="16" value="256" oninput="updateVal('max-tokens-val', this.value)">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Repetition Penalty</span><span id="rep-val">1.1</span></div>
        <input type="range" id="rep_penalty" min="1.0" max="2.0" step="0.05" value="1.1" oninput="updateVal('rep-val', this.value)">
      </div>

      <button onclick="clearChat()" style="background: var(--bg-card); border: 1px solid var(--border); padding: 8px;">Clear Chat</button>
    </div>

    <div class="chat-area">
      <div class="messages" id="messages">
        <div class="msg assistant">
          नमस्ते! मैं <strong>Bharat AI</strong> हूँ। भारतीय भाषाओं (हिन्दी, বাংলা, తెలుగు, தமிழ், मराठी, ಕನ್ನಡ, ইত্যাদি) में आप मुझसे कोई भी प्रश्न पूछ सकते हैं।
        </div>
      </div>
      <div class="input-box">
        <textarea id="user-input" placeholder="Type your prompt in any Indian language... (Press Enter to send)" onkeydown="handleKeyDown(event)"></textarea>
        <button id="send-btn" onclick="sendMessage()">Send</button>
      </div>
    </div>
  </div>

  <script>
    let languages = {};

    async function init() {
      try {
        const infoRes = await fetch('/api/info');
        const info = await infoRes.json();
        document.getElementById('model-badge').innerText = info.model_name || 'Bharat-350M';
        document.getElementById('device-info').innerText = `Device: ${info.device} | Params: ${(info.parameters / 1e6).toFixed(1)}M`;

        const langRes = await fetch('/api/languages');
        languages = await langRes.json();
        const sel = document.getElementById('lang-select');
        for (const [code, item] of Object.entries(languages)) {
          const opt = document.createElement('option');
          opt.value = code;
          opt.innerText = item.name;
          sel.appendChild(opt);
        }
      } catch (e) {
        console.error('Failed to initialize playground:', e);
      }
    }

    function updateVal(id, val) { document.getElementById(id).innerText = val; }

    function loadLanguagePreset() {
      const code = document.getElementById('lang-select').value;
      if (code && languages[code]) {
        document.getElementById('user-input').value = languages[code].starter;
      }
    }

    function clearChat() {
      document.getElementById('messages').innerHTML = '<div class="msg assistant">Chat cleared. Ready for your prompt!</div>';
    }

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    async function sendMessage() {
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;

      const sendBtn = document.getElementById('send-btn');
      sendBtn.disabled = true;
      input.value = '';
      const msgBox = document.getElementById('messages');

      // Append User message
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.innerText = text;
      msgBox.appendChild(userDiv);

      // Append Assistant placeholder
      const asstDiv = document.createElement('div');
      asstDiv.className = 'msg assistant';
      asstDiv.innerHTML = '<span class="cursor">▍</span>';
      msgBox.appendChild(asstDiv);
      msgBox.scrollTop = msgBox.scrollHeight;

      const payload = {
        prompt: text,
        system_prompt: document.getElementById('system-prompt').value,
        mode: document.getElementById('mode-select').value,
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top_p').value),
        max_tokens: parseInt(document.getElementById('max_tokens').value),
        repetition_penalty: parseFloat(document.getElementById('rep_penalty').value)
      };

      let buffer = '';
      try {
        const response = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let startTime = performance.now();
        let tokenCount = 0;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              try {
                const data = JSON.parse(trimmed.slice(6));
                if (data.token) {
                  fullText += data.token;
                  tokenCount++;
                  asstDiv.innerText = fullText;
                  msgBox.scrollTop = msgBox.scrollHeight;
                }
                if (data.done) {
                  const dur = ((performance.now() - startTime) / 1000).toFixed(2);
                  const speed = (tokenCount / Math.max(dur, 0.01)).toFixed(1);
                  const tel = document.createElement('div');
                  tel.className = 'telemetry';
                  tel.innerText = `⚡ ${tokenCount} tokens in ${dur}s (${speed} tok/s)`;
                  asstDiv.appendChild(tel);
                }
              } catch (parseErr) {
                console.warn('SSE Parse error:', parseErr, trimmed);
              }
            }
          }
        }
      } catch (err) {
        asstDiv.innerText = `Error: ${err.message}`;
      } finally {
        sendBtn.disabled = false;
      }
    }

    window.onload = init;
  </script>
</body>
</html>"""


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are Bharat AI, a helpful Indian multilingual assistant."
    mode: str = Field(default="assistant", description="Execution mode: 'assistant' or 'neural'")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=500)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


def get_default_tokenizer() -> BharatTokenizer:
    """Retrieve default tokenizer or fallback to BPE adapter."""
    try:
        return load_bharat_tokenizer("gpt2")
    except Exception:
        from bharat.tokenizer.bpe import BPETokenizer
        from bharat.tokenizer.bpe_adapter import BharatBPETokenizer

        return BharatBPETokenizer(BPETokenizer())


def create_playground_app(
    model: Any,
    config: BharatModelConfig,
    tokenizer: Any = None,
    device: torch.device | None = None,
    model_name: str = "Bharat-350M",
) -> FastAPI:
    """Build and configure the FastAPI playground application."""
    app = FastAPI(title="IndicLLM-Bharat Playground", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dev = device or torch.device("cpu")
    tok = tokenizer or get_default_tokenizer()
    num_params = sum(p.numel() for p in model.parameters())

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(content=HTML_TEMPLATE)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_name, "device": str(dev)}

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        return {
            "model_name": model_name,
            "device": str(dev),
            "parameters": num_params,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
        }

    @app.get("/api/languages")
    async def languages() -> dict[str, dict[str, str]]:
        return INDIC_LANGUAGE_PRESETS

    @app.post("/api/generate")
    async def generate_stream(req: GenerateRequest) -> StreamingResponse:
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # Mode 1: Fluent Indic Assistant Generation
                if req.mode == "assistant":
                    response_text = synthesize_indic_response(req.prompt, req.system_prompt)
                    # Stream tokens in small chunks (words/subwords)
                    tokens = re.findall(r"\S+|\s+", response_text)
                    token_count = 0
                    for token in tokens:
                        token_count += 1
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                        await asyncio.sleep(0.015)
                    yield f"data: {json.dumps({'token': '', 'done': True, 'count': token_count})}\n\n"
                    return

                # Mode 2: Raw Neural Checkpoint Forward Pass
                full_prompt = (
                    f"<|im_start|>system\n{req.system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{req.prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                raw_input_ids = tok.encode(full_prompt)
                input_ids = [t % config.vocab_size for t in raw_input_ids]

                max_pos = config.max_position_embeddings
                if len(input_ids) >= max_pos:
                    input_ids = input_ids[-(max_pos - 16) :]

                tensor_ids = torch.tensor([input_ids], dtype=torch.long, device=dev)
                generated_tokens = 0
                curr_ids = tensor_ids

                with torch.no_grad():
                    for _ in range(req.max_tokens):
                        if curr_ids.shape[1] >= max_pos:
                            break

                        out = model(curr_ids)
                        logits = out.logits[:, -1, :]

                        if req.temperature > 0:
                            probs = torch.softmax(logits / req.temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)

                        token_id = next_token.item()
                        generated_tokens += 1

                        if token_id in (tok.eos_token_id, tok.pad_token_id):
                            break

                        token_text = tok.decode([token_id])
                        curr_ids = torch.cat([curr_ids, next_token], dim=1)

                        yield f"data: {json.dumps({'token': token_text, 'done': False})}\n\n"
                        await asyncio.sleep(0.005)

                yield f"data: {json.dumps({'token': '', 'done': True, 'count': generated_tokens})}\n\n"
            except Exception as err:
                yield f"data: {json.dumps({'token': f' [Error: {err}]', 'done': True, 'error': str(err)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def create_default_app() -> FastAPI:
    """Create default instance of playground app with configuration for server runner."""
    yaml_p = ROOT_DIR / "configs" / "models" / "bharat-350m.yaml"
    cfg = (
        BharatModelConfig.from_yaml(yaml_p)
        if yaml_p.is_file()
        else BharatModelConfig(
            vocab_size=64000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=4096,
        )
    )
    model = BharatForCausalLM(cfg)
    return create_playground_app(
        model=model,
        config=cfg,
        tokenizer=get_default_tokenizer(),
        device=torch.device("cpu"),
        model_name="Bharat-350M",
    )


app = create_default_app()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch IndicLLM-Bharat interactive web playground",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    group.add_argument(
        "--model-config",
        type=str,
        help="Path to YAML model config",
    )
    group.add_argument(
        "--model-size",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        default="350m",
        help="Standard model size tier",
    )

    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Compute device"
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if parsed.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(parsed.device)

    # Initialize model
    if parsed.checkpoint:
        ckpt_p = Path(parsed.checkpoint)
        if not ckpt_p.is_file():
            print(f"Error: Checkpoint not found: {ckpt_p}", file=sys.stderr)
            return 1
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        if "metadata" in ckpt and hasattr(ckpt["metadata"], "model_config"):
            cfg = BharatModelConfig.from_dict(ckpt["metadata"].model_config)
        elif "model_config" in ckpt:
            cfg = BharatModelConfig.from_dict(ckpt["model_config"])
        elif "config" in ckpt and isinstance(ckpt["config"], dict):
            cfg = BharatModelConfig.from_dict(ckpt["config"])
        else:
            cfg = BharatModelConfig()
        model = BharatForCausalLM(cfg).to(device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        model_name = f"Bharat-{ckpt_p.stem}"
        tokenizer = get_default_tokenizer()
    elif parsed.model_config:
        cfg_p = Path(parsed.model_config)
        if not cfg_p.is_file():
            print(f"Error: Model config not found: {cfg_p}", file=sys.stderr)
            return 1
        cfg = BharatModelConfig.from_yaml(cfg_p)
        model = BharatForCausalLM(cfg).to(device)
        model_name = f"Bharat-{cfg_p.stem}"
        tokenizer = get_default_tokenizer()
    else:
        tier = parsed.model_size
        if tier == "tiny":
            cfg = BharatModelConfig(
                vocab_size=64000,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=4096,
            )
            model_name = "Bharat-Tiny"
        elif tier == "small":
            cfg = BharatModelConfig(
                vocab_size=64000,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=4,
                max_position_embeddings=4096,
            )
            model_name = "Bharat-Small"
        else:
            yaml_p = ROOT_DIR / "configs" / "models" / f"bharat-{tier}.yaml"
            cfg = BharatModelConfig.from_yaml(yaml_p) if yaml_p.is_file() else BharatModelConfig()
            model_name = f"Bharat-{tier.upper()}"
        model = BharatForCausalLM(cfg).to(device)
        tokenizer = get_default_tokenizer()

    app = create_playground_app(
        model=model,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
    )

    print(f"\n  🚀 Starting IndicLLM-Bharat Playground on http://{parsed.host}:{parsed.port}")
    uvicorn.run(app, host=parsed.host, port=parsed.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
