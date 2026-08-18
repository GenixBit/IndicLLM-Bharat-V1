"""High-density Multilingual and Technical Preference Dataset Generator for IndicLLM-Bharat.

Generates structured chosen vs rejected preference pairs across all 22 Scheduled
Indian Languages and worldwide science, mathematics, computer science, and modern AI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INDIC_PREFERENCES: list[dict[str, Any]] = [
    # 1. Hindi (hi)
    {
        "lang": "hi",
        "domain": "indic_culture_history",
        "prompt": "भारत के संविधान की मुख्य विशेषताएँ क्या हैं?",
        "chosen": (
            "भारतीय संविधान की प्रमुख विशेषताएँ निम्नलिखित हैं:\n\n"
            "1. **लिखित एवं विस्तृत संविधान**: यह विश्व का सबसे बड़ा हस्तलिखित एवं विस्तृत लिखित संविधान है।\n"
            "2. **संप्रभु, समाजवादी, पंथनिरपेक्ष, लोकतांत्रिक गणराज्य**: प्रस्तावना भारत को एक स्वतंत्र एवं समावेशी राष्ट्र घोषित करती है।\n"
            "3. **मौलिक अधिकार एवं कर्तव्य**: भाग ३ में ६ मौलिक अधिकार और भाग ४क में मौलिक कर्तव्य दिए गए हैं।\n"
            "4. **स्वतंत्र न्यायपालिका एवं त्रिस्तरीय शासन**: केंद्र, राज्य और स्थानीय स्तर पर शक्तियों का स्पष्ट विभाजन है।"
        ),
        "rejected": (
            "संविधान बस एक नियम की किताब है। इसमें बहुत से नियम हैं जो सरकार बनाती है। "
            "सबको कानून मानना पड़ता है नहीं तो सजा मिलती है।"
        ),
    },
    {
        "lang": "hi",
        "domain": "indic_literature",
        "prompt": "मुंशी प्रेमचंद के प्रमुख उपन्यासों के नाम बताइए।",
        "chosen": (
            "मुंशी प्रेमचंद (१८८०–१९३६) हिन्दी एवं उर्दू के मूर्धन्य उपन्यासकार हैं। उनके प्रमुख उपन्यास हैं:\n\n"
            "1. **गोदान (१९३६)**: भारतीय किसान जीवन एवं सामाजिक यथार्थ का महाकाव्य।\n"
            "2. **गबन (१९३१)**: मध्यवर्गीय नैतिकता और आभूषण-लोलुपता की मनोवैज्ञानिक त्रासदी।\n"
            "3. **कर्मभूमि (१९३२)**: स्वतंत्रता आंदोलन एवं सामाजिक सुधार का चित्रण।\n"
            "4. **रंगभूमि एवं सेवासदन**: जनआंदोलनों और नारी-उत्थान की सशक्त अभिव्यक्ति।"
        ),
        "rejected": (
            "मुंशी प्रेमचंद ने बहुत सी किताबें लिखी हैं जैसे गोदान, गीतांजलि, और कालिदास के नाटक। "
            "वे बहुत बड़े लेखक थे और कविताएँ भी लिखते थे।"
        ),
    },
    # 2. Bengali (bn)
    {
        "lang": "bn",
        "domain": "indic_literature",
        "prompt": "রবীন্দ্রনাথ ঠাকুরের সাহিত্যে অবদানের সংক্ষেপ দিন।",
        "chosen": (
            "রবীন্দ্রনাথ ঠাকুর (১৮৬১-১৯৪১) ছিলেন বিশ্ববরেণ্য কবি, ঔপন্যাসিক, সংগীতজ্ঞ ও দার্শনিক:\n\n"
            "১. **নোবেল পুরস্কার (১৯১৩)**: গীতাঞ্জলি কাব্যগ্রন্থের জন্য সাহিত্যে প্রথম এশীয় নোবেল জয়।\n"
            "২. **জাতীয় সংগীত**: ভারত ('জন গণ মন') ও বাংলাদেশ ('আমার সোনার বাংলা')-এর জাতীয় সংগীত রচয়িতা।\n"
            "৩. **বিশ্বভারতী প্রতিষ্ঠা**: শান্তি-নিকেতনে উন্মুক্ত প্রকৃতি ও সার্বজনীন শিক্ষার মেলবন্ধন।"
        ),
        "rejected": (
            "রবীন্দ্রনাথ ঠাকুর কলকাতার একজন সাধারণ কবি ছিলেন। তিনি কিছু গান এবং কবিতা লিখেছিলেন "
            "এবং শান্তিনিকেতনে থাকতেন।"
        ),
    },
    # 3. Tamil (ta)
    {
        "lang": "ta",
        "domain": "indic_literature",
        "prompt": "திருக்குறளின் சிறப்பம்சங்களை விவரிக்கவும்.",
        "chosen": (
            "திருவள்ளுவர் இயற்றிய **திருக்குறள்** உலகப் பொதுமறையாகப் போற்றப்படுகிறது:\n\n"
            "1. **முப்பால் பகுப்பு**: அறத்துப்பால் (380), பொருட்பால் (700), காமத்துப்பால் (250) என மொத்தம் 1330 குறள்கள்.\n"
            "2. **மதச்சார்பற்ற அறநெறி**: எந்தவொரு குறிப்பிட்ட மதம் அல்லது இனத்தை சாராமல் உலக மக்கள் அனைவருக்கும் பொதுவான நன்னெறிகளைப் புகட்டுகிறது.\n"
            "3. **சுருக்கமும் ஆழமும்**: இரண்டடி குறள் வெண்பாவில் உலகளாவிய மெய்யியலைத் தெளிவாக விளக்குகிறது."
        ),
        "rejected": (
            "திருக்குறள் ஒரு பழைய தமிழ் புத்தகம். அதில் நிறைய செய்யுள்கள் உள்ளன. மக்கள் அதை படித்து "
            "நல்லவர்களாக இருக்க வேண்டும் என்று சொல்கிறது."
        ),
    },
    # 4. Telugu (te)
    {
        "lang": "te",
        "domain": "indic_geography",
        "prompt": "ఆంధ్రప్రదేశ్ మరియు తెలంగాణ ప్రధాన నదుల గురించి వివరించండి.",
        "chosen": (
            "ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాల గుండా ప్రవహించే ముఖ్యమైన నదులు:\n\n"
            "1. **గోదావరి (దక్షిణ గంగ)**: మహారాష్ట్రలోని త్రయంబకేశ్వర్ వద్ద జన్మించి, తెలంగాణ (బాసర, కాళేశ్వరం) మరియు ఆంధ్రప్రదేశ్ (రాజమండ్రి) మీదుగా బంగాళాఖాతంలో కలుస్తుంది.\n"
            "2. **కృష్ణా నది**: మహాబలేశ్వర్ వద్ద ఉద్భవించి, నాగార్జున సాగర్ మరియు ప్రకాశం బ్యారేజ్ ద్వారా సస్యశ్యామలం చేస్తుంది.\n"
            "3. **తుంగభద్ర & పెన్నా**: రాయలసీమ మరియు దక్కన్ పీఠభూమికి జీవనాధారమైన ఉపనదులు."
        ),
        "rejected": (
            "తెలంగాణ మరియు ఆంధ్రాలో చాలా నదులు ఉన్నాయి. గోదావరి మరియు గంగా నది అక్కడ ప్రవహిస్తాయి "
            "మరియు నీరు చాలా ఎక్కువగా ఉంటుంది."
        ),
    },
    # 5. Marathi (mr)
    {
        "lang": "mr",
        "domain": "indic_history",
        "prompt": "छत्रपती शिवाजी महाराजांच्या हिंदवी स्वराज्याची प्रशासकीय वैशिष्ट्ये कोणती?",
        "chosen": (
            "छत्रपती शिवाजी महाराजांचे (१६३०-१६८०) स्वराज्य हे लोककल्याणकारी व कार्यक्षम प्रशासनाचे आदर्श प्रतीक होते:\n\n"
            "१. **अष्टप्रधान मंडळ**: प्रशासनाची विभागणी ८ महत्त्वाच्या खात्यांमध्ये करून जबाबदाऱ्यांचे वाटप.\n"
            "२. **रयतेचे संरक्षण**: शेतकऱ्यांच्या पिकांचे नुकसान न करणे, करप्रणालीत पारदर्शकता व न्याय्य महसूल.\n"
            "३. **किल्ले व आरमार**: सागरी सीमांच्या रक्षणासाठी सिंधुदुर्ग, विजयदुर्ग यांसारखे अजिंक्य जलदुर्ग व आधुनिक आरमाराची उभारणी.\n"
            "४. **गनिमी कावा (शिवसूत्र)**: भौगोलिक परिस्थितीचा पुरेपूर वापर करून शत्रूवर मात करण्याचे युद्धतंत्र."
        ),
        "rejected": (
            "शिवाजी महाराज एक राजे होते आणि त्यांनी खूप लढाया जिंकल्या. त्यांच्याकडे खूप किल्ले होते "
            "आणि सैन्य होते."
        ),
    },
    # 6. Gujarati (gu)
    {
        "lang": "gu",
        "domain": "indic_history",
        "prompt": "મહાત્મા ગાંધીજીના સત્યાગ્રહ આંદોલનની મુખ્ય પદ્ધતિઓ સમજાવો.",
        "chosen": (
            "મહાત્મા ગાંધીજીના સત્યાગ્રહના મુખ્ય સિદ્ધાંતો નીચે મુજબ છે:\n\n"
            "૧. **અહિંસા (Non-Violence)**: મન, વચન અને કર્મથી કોઈને પણ ઈજા ન પહોંચાડવી.\n"
            "૨. **સત્ય (Truth)**: ન્યાય અને સત્ય માટે કોઈપણ ભય વગર દ્રઢપણે ઊભા રહેવું.\n"
            "૩. **સવિનય કાનૂન ભંગ**: અન્યાયી કાયદાઓનો શાંતિપૂર્ણ અને અહિંસક રીતે વિરોધ કરવો (દા.ત. દાંડી કૂચ ૧૯૩૦).\n"
            "૪. **સ્વદેશી અને બહિષ્કાર**: ખાદીનો પ્રચાર અને વિદેશી વસ્તુઓનો બહિષ્કાર કરી આત્મનિર્ભરતા કેળવવી."
        ),
        "rejected": (
            "ગાંધીજીએ આઝાદીની લડાઈ લડી હતી. તેમણે ઉપવાસ કર્યા હતા અને લોકોને અંગ્રેજો સામે લડવા કહ્યું હતું."
        ),
    },
    # 7. Kannada (kn)
    {
        "lang": "kn",
        "domain": "indic_literature",
        "prompt": "ಕನ್ನಡ ಸಾಹಿತ್ಯದ ವಚನ ಚಳವಳಿಯ ಮಹತ್ವವೇನು?",
        "chosen": (
            "೧೨ನೇ ಶತಮಾನದ **ವಚನ ಚಳವಳಿ** ಕನ್ನಡ ಸಾಹಿತ್ಯ ಮತ್ತು ಸಮಾಜ ಸುಧಾರಣೆಯ ಮಹತ್ವದ ಕ್ರಾಂತಿ:\n\n"
            "೧. **ಬಸವೇಶ್ವರರು ಮತ್ತು ಶರಣರು**: ಅಲ್ಲಮಪ್ರಭು, ಅಕ್ಕಮಹಾದೇವಿ, ಚನ್ನಬಸವಣ್ಣ ಮುಂತಾದ ಶರಣರು ಸರಳ ಕನ್ನಡದಲ್ಲಿ ವಚನಗಳನ್ನು ರಚಿಸಿದರು.\n"
            "೨. **ಅನುಭವ ಮಂಟಪ**: ವಿಶ್ವದ ಮೊದಲ ಪ್ರಜಾಪ್ರಭುತ್ವ ಮಾದರಿಯ ಸಂಸತ್ತು, ಸಮಾನತೆ ಮತ್ತು ಕಾಯಕ ತತ್ವದ ಪ್ರತಿಪಾದನೆ.\n"
            "೩. **ಸಾಮಾಜಿಕ ಸಮಾನತೆ**: ಜಾತಿ, ಲಿಂಗ ಮತ್ತು ವರ್ಗ ಭೇದಗಳನ್ನು ನಿರಾಕರಿಸಿ 'ಕಾಯಕವೇ ಕೈಲಾಸ' ಎಂಬ ಸಂದೇಶ ಸಾರಿದರು."
        ),
        "rejected": (
            "ವಚನ ಚಳವಳಿ ಎಂದರೆ ಹಾಡುಗಳನ್ನು ಹಾಡುವುದು. ಬಸವಣ್ಣನವರು ಕೆಲವು ಪದ್ಯಗಳನ್ನು ಬರೆದಿದ್ದರು "
            "ಮತ್ತು ಜನರಿಗೆ ಒಳ್ಳೆಯವರಾಗಿರಲು ಹೇಳಿದರು."
        ),
    },
    # 8. Malayalam (ml)
    {
        "lang": "ml",
        "domain": "indic_culture",
        "prompt": "കേരളത്തിന്റെ പരമ്പരാഗത കലാരൂപങ്ങളെക്കുറിച്ച് വിവരിക്കുക.",
        "chosen": (
            "കേരളത്തിന്റെ സമ്പന്നമായ സാംസ്കാരിക കലാരൂപങ്ങൾ:\n\n"
            "1. **കഥകളി**: വേഷഭൂഷകളും മുദ്രകളും മുഖഭാവങ്ങളും കൊണ്ട് കഥ പറയുന്ന ക്ലാസിക്കൽ നൃത്തരൂപം.\n"
            "2. **മോഹിനിയാട്ടം**: സ്ത്രീകളുടെ ലാസ്യഭാവ പ്രധാനമായ പരമ്പരാഗത നൃത്തം.\n"
            "3. **തെയ്യം & കൂടിയാട്ടം**: യുനെസ്കോ പൈതൃക പട്ടികയിലുള്ള പ്രാചീന അനുഷ്ഠാന നാട്യകല.\n"
            "4. **കളരിപ്പയറ്റ്**: ലോകത്തിലെ തന്നെ ഏറ്റവും പുരാതനമായ ആയോധനകല."
        ),
        "rejected": ("കേരളത്തിൽ കഥകളി ഉണ്ട്. ആളുകൾ വലിയ വസ്ത്രം ധരിച്ച് നൃത്തം ചെയ്യും. " "ഇത് വളരെ പഴയ കലയാണ്."),
    },
    # 9. Punjabi (pa)
    {
        "lang": "pa",
        "domain": "indic_history",
        "prompt": "ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ ਦੇ ਤਿੰਨ ਮੁੱਖ ਉਪਦੇਸ਼ ਕਿਹੜੇ ਹਨ?",
        "chosen": (
            "ਸ੍ਰੀ ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ (੧੪੬੯–੧੫੩੯) ਨੇ ਮਨੁੱਖਤਾ ਲਈ ਤਿੰਨ ਬੁਨਿਆਦੀ ਸਿਧਾਂਤ ਦਿੱਤੇ:\n\n"
            "੧. **ਨਾਮ ਜਪੋ**: ਪਰਮਾਤਮਾ ਦੇ ਨਾਮ ਦਾ ਸਿਮਰਨ ਕਰਨਾ ਅਤੇ ਹਰ ਪਲ ਉਸ ਦੀ ਰਜ਼ਾ ਵਿੱਚ ਰਹਿਣਾ।\n"
            "੨. **ਕਿਰਤ ਕਰੋ**: ਸੱਚੀ-ਸੁੱਚੀ, ਇਮਾਨਦਾਰੀ ਅਤੇ ਮਿਹਨਤ ਨਾਲ ਰੋਜ਼ੀ-ਰੋਟੀ ਕਮਾਉਣੀ।\n"
            "੩. **ਵੰਡ ਛਕੋ**: ਲੋੜਵੰਦਾਂ ਨਾਲ ਆਪਣੀ ਕਮਾਈ ਸਾਂਝੀ ਕਰਨੀ ਅਤੇ ਲੰਗਰ-ਸੇਵਾ ਰਾਹੀਂ ਬਰਾਬਰੀ ਕਾਇਮ ਕਰਨੀ।"
        ),
        "rejected": ("ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ ਨੇ ਕਿਹਾ ਸੀ ਕਿ ਰੱਬ ਇੱਕ ਹੈ ਅਤੇ ਸਭ ਨੂੰ ਆਪਸ ਵਿੱਚ ਪਿਆਰ ਨਾਲ ਰਹਿਣਾ ਚਾਹੀਦਾ ਹੈ।"),
    },
    # 10. Odia (or)
    {
        "lang": "or",
        "domain": "indic_culture",
        "prompt": "ପୁରୀ ଜଗନ୍ନାଥ ମନ୍ଦିର ଏବଂ ରଥଯାତ୍ରାର ମହତ୍ତ୍ୱ କ'ଣ?",
        "chosen": (
            "ପୁରୀର ଶ୍ରୀଜଗନ୍ନାଥ ମନ୍ଦିର ଏବଂ ବିଶ୍ୱପ୍ରସିଦ୍ଧ ରଥଯାତ୍ରା ଭାରତୀୟ ସଂସ୍କୃତିର ଅନନ୍ୟ ପ୍ରତୀକ:\n\n"
            "୧. **ଚତୁର୍ଦ୍ଧା ମୂର୍ତ୍ତି**: ପ୍ରଭୁ ଜଗନ୍ନାଥ, ବଳଭଦ୍ର, ଦେବୀ ସୁଭଦ୍ରା ଏବଂ ସୁଦର୍ଶନଙ୍କ ଦାରୁମୂର୍ତ୍ତି ପୂଜା ପାଆନ୍ତି।\n"
            "୨. **ମହାପ୍ରସାଦ ଓ ସମାନତା**: ଆନନ୍ଦ ବଜାରରେ ଜାତି-ଧର୍ମ ନିର୍ବିଶେଷରେ ସମସ୍ତେ ଏକାଠି ମହାପ୍ରସାଦ ଗ୍ରହଣ କରନ୍ତି।\n"
            "୩. **ରଥଯାତ୍ରା**: ପ୍ରଭୁ ନିଜେ ରତ୍ନବେଦୀ ଛାଡ଼ି ବଡ଼ଦାଣ୍ଡକୁ ଆସି ସମସ୍ତ ଭକ୍ତଙ୍କୁ ଦର୍ଶନ ଦିଅନ୍ତି।"
        ),
        "rejected": ("ପୁରୀରେ ଏକ ବଡ଼ ମନ୍ଦିର ଅଛି ଯେଉଁଠି ପ୍ରତିବର୍ଷ ରଥଯାତ୍ରା ହୁଏ ଏବଂ ବହୁତ ଲୋକ ଆସନ୍ତି।"),
    },
    # 11. Assamese (as)
    {
        "lang": "as",
        "domain": "indic_culture",
        "prompt": "অসমৰ বিহু উৎসৱৰ তিনিটা প্ৰকাৰ কি কি?",
        "chosen": (
            "অসমৰ জাতীয় উৎসৱ বিহু তিনিটা ঋতুত তিনিটা ৰূপত উদযাপিত হয়:\n\n"
            "১. **ৰঙালী বা ব'হাগ বিহু (এপ্ৰিল)**: অসমীয়া নৱবৰ্ষ আৰু বসন্তকালীন আনন্দ-নৃত্যৰ উৎসৱ।\n"
            "২. **কঙালী বা কাতি বিহু (অক্টোবৰ)**: শস্যৰ শ্ৰীবৃদ্ধি আৰু মংগলৰ বাবে তুলসী তলত চাকি জ্বলোৱাৰ শান্ত উৎসৱ।\n"
            "३. **ভোগালী বা মাঘ বিহু (জানুৱাৰী)**: শস্য চপোৱাৰ আনন্দ, মেজি আৰু ভেলাঘৰ সাজি ভোজ খোৱাৰ উৎসৱ।"
        ),
        "rejected": ("বিহু অসমৰ উৎসৱ। ইয়াত বিহু নৃত্য কৰা হয় আৰু পিঠা খোৱা হয়।"),
    },
    # 12. Urdu (ur)
    {
        "lang": "ur",
        "domain": "indic_literature",
        "prompt": "مرزا غالب کی شاعری کی اہم خصوصیات کیا ہیں؟",
        "chosen": (
            "مرزا اسد اللہ خان غالب (۱۷۹۷–۱۸۶۹) اردو زبان و ادب کے عظیم ترین شاعر ہیں:\n\n"
            "۱. **فلسفیانہ فکر**: انسانی وجود، تقدیر اور کائنات کے اسرار پر گہری سوچ۔\n"
            "۲. **جدتِ طراز اسلوب**: روایتی تشبیہات سے ہٹ کر نئے استعارے اور لطیف شوخی۔\n"
            "۳. **خطوطِ غالب**: اردو نثری اسلوب کو آسان اور گفتگو کی مانند دلکش بنا دیا۔"
        ),
        "rejected": (
            "غالب دہلی کے شاعر تھے اور وہ بہت اچھی غزلیں لکھتے تھے جو سب کو پسند آتی تھیں۔"
        ),
    },
    # 13. Sanskrit (sa)
    {
        "lang": "sa",
        "domain": "indic_philosophy",
        "prompt": "श्रीमद्भगवद्गीतायाः निष्कामकर्मयोगस्य तात्पर्यं किम्?",
        "chosen": (
            "श्रीमद्भगवद्गीतायां श्रीकृष्णः निष्कामकर्मयोगस्य अद्वितीयं सिद्धान्तं प्रतिपादयति:\n\n"
            "**'कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥'** (२.४७)\n\n"
            "१. **कर्तव्यनिष्ठा**: फलासक्तिं विहाय स्वधर्मस्य सम्यक् पालनम्।\n"
            "२. **चित्तशुद्धिः**: कर्मफले समत्वबुद्धिरेव योगः उच्यते ('समत्वं योग उच्यते')।"
        ),
        "rejected": ("गीता एका धार्मिक पुस्तक अस्ति यत्र अर्जुनस्य युद्धस्य कथा अस्ति।"),
    },
]

STEM_AND_GLOBAL_PREFERENCES: list[dict[str, Any]] = [
    # 1. AI & Machine Learning
    {
        "domain": "modern_ai",
        "prompt": "Compare Grouped-Query Attention (GQA) with Multi-Head Attention (MHA) and Multi-Query Attention (MQA).",
        "chosen": (
            "Attention mechanisms in transformer architectures trade off expressive capability against inference memory bandwidth:\n\n"
            "1. **Multi-Head Attention (MHA)**:\n"
            "   - Query, Key, and Value heads are 1:1 ($H_q = H_k = H_v$).\n"
            "   - Highest expressive capacity, but largest KV-cache memory requirement during autoregressive generation.\n\n"
            "2. **Multi-Query Attention (MQA)**:\n"
            "   - All query heads share a single Key and Value head ($H_k = H_v = 1$).\n"
            "   - Extremely small KV-cache footprint, but can suffer quality and training stability degradation.\n\n"
            "3. **Grouped-Query Attention (GQA)**:\n"
            "   - Groups multiple query heads per Key/Value head (e.g., 4:1 or 8:1 ratio, where $H_q / H_{kv} = G$).\n"
            "   - Interpolates between MHA and MQA, delivering near-MHA perplexity with up to 8x lower KV-cache bandwidth demand (utilized in IndicLLM-Bharat)."
        ),
        "rejected": (
            "GQA, MHA, and MQA are all attention mechanisms in transformers. MHA has heads, MQA has one head, "
            "and GQA has groups. They help the model remember things faster."
        ),
    },
    # 2. Quantum Computing
    {
        "domain": "quantum_physics",
        "prompt": "What is the difference between quantum superposition and quantum entanglement?",
        "chosen": (
            "Both are foundational quantum phenomena with distinct definitions and mathematical formulations:\n\n"
            "1. **Quantum Superposition (Single-qubit property)**:\n"
            "   - A single quantum state can exist as a linear combination of basis states: $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$, where $|\\alpha|^2 + |\\beta|^2 = 1$.\n"
            "   - Enables evaluating an exponential state space ($2^n$) across $n$ qubits simultaneously.\n\n"
            "2. **Quantum Entanglement (Multi-particle composite property)**:\n"
            "   - Two or more particles become correlated such that the total state cannot be factored into individual states: e.g., the Bell state $|\\Phi^+\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)$.\n"
            "   - Measuring one qubit instantaneously determines the state of the other, regardless of spatial separation.\n\n"
            "- **Summary**: Superposition describes coexistence of multiple states in a single system; Entanglement describes non-local correlation between multiple systems."
        ),
        "rejected": (
            "Superposition means particles are everywhere at once, and entanglement is Einstein's spooky action "
            "at a distance where they talk to each other across space."
        ),
    },
    # 3. Computer Science / Algorithms
    {
        "domain": "computer_science_algorithms",
        "prompt": "Write a clean Python function for finding the Longest Substring Without Repeating Characters and explain its time complexity.",
        "chosen": (
            "Here is the optimal **Sliding Window** implementation using a hash map in Python:\n\n"
            "```python\n"
            "def length_of_longest_substring(s: str) -> int:\n"
            '    """Find length of longest substring without repeating characters in O(n) time."""\n'
            "    char_index_map: dict[str, int] = {}\n"
            "    max_len = 0\n"
            "    start = 0\n\n"
            "    for end, char in enumerate(s):\n"
            "        if char in char_index_map and char_index_map[char] >= start:\n"
            "            start = char_index_map[char] + 1\n"
            "        char_index_map[char] = end\n"
            "        max_len = max(max_len, end - start + 1)\n\n"
            "    return max_len\n\n"
            "# Example\n"
            "print(length_of_longest_substring('abcabcbb'))  # Output: 3 ('abc')\n"
            "print(length_of_longest_substring('bbbbb'))     # Output: 1 ('b')\n"
            "```\n\n"
            "- **Time Complexity**: $O(n)$ where $n$ is string length (single pass).\n"
            "- **Space Complexity**: $O(\\min(n, |\\Sigma|))$ for the hash map of character indices."
        ),
        "rejected": (
            "```python\n"
            "def lengthOfLongestSubstring(s):\n"
            "    return len(set(s))\n"
            "```\n"
            "This checks unique characters and returns the set length."
        ),
    },
    # 4. Mathematics / Calculus
    {
        "domain": "mathematics_calculus",
        "prompt": "State and explain the Fundamental Theorem of Calculus with an example.",
        "chosen": (
            "The **Fundamental Theorem of Calculus (FTC)** establishes the link between differentiation and integration:\n\n"
            "1. **First Part (FTC-1)**:\n"
            "   If $f$ is continuous on $[a, b]$ and $g(x) = \\int_a^x f(t)dt$, then $g'(x) = f(x)$. (Differentiation undoes integration).\n\n"
            "2. **Second Part (FTC-2 - Evaluation Theorem)**:\n"
            "   \\[\n"
            "   \\int_a^b f(x)\\,dx = F(b) - F(a)\n"
            "   \\]\n"
            "   where $F$ is any antiderivative of $f$ ($F'(x) = f(x)$).\n\n"
            "**Example**:\n"
            "Evaluate $\\int_0^3 2x\\,dx$:\n"
            "- Antiderivative: $F(x) = x^2$\n"
            "- Calculation: $F(3) - F(0) = 3^2 - 0^2 = 9$."
        ),
        "rejected": (
            "Calculus theorem says integration is adding things up and differentiation is finding slopes. "
            "You just take the integral of f(x) and subtract the numbers."
        ),
    },
    # 5. Space & Technology (ISRO)
    {
        "domain": "space_technology",
        "prompt": "What were the primary scientific instruments and discoveries of the Chandrayaan-3 mission?",
        "chosen": (
            "**Chandrayaan-3** successfully landed near the lunar South Pole on August 23, 2023:\n\n"
            "1. **Vikram Lander Payloads**:\n"
            "   - **ChaSTE** (Chandra's Surface Thermophysical Experiment): Measured thermal gradient of lunar regolith, revealing a sharp drop from 50°C at surface to -10°C at 8 cm depth.\n"
            "   - **ILSA** (Instrument for Lunar Seismic Activity): Recorded lunar seismic events and micrometeroid impacts.\n"
            "   - **RAMBHA-LP**: Measured lunar plasma density.\n\n"
            "2. **Pragyan Rover Payloads**:\n"
            "   - **LIBS** (Laser-Induced Breakdown Spectroscopy) & **APXS**: Confirmed unambiguous presence of **Sulphur (S)**, along with Al, Ca, Fe, Cr, and Ti in the South Polar regolith."
        ),
        "rejected": (
            "Chandrayaan-3 went to the moon and landed safely. The rover drove around taking pictures "
            "and found that the moon has rocks and dust."
        ),
    },
    # 6. Macroeconomics & Digital Public Infrastructure
    {
        "domain": "economics_dpi",
        "prompt": "How does India's Digital Public Infrastructure (India Stack) work and what are its core layers?",
        "chosen": (
            "**India Stack** is a set of open APIs and digital public goods built across three foundational layers:\n\n"
            "1. **Identity Layer (Aadhaar)**:\n"
            "   - Biometric-backed digital identity for 1.4B+ citizens enabling paperless e-KYC and digital authentication.\n\n"
            "2. **Payments Layer (UPI & IMPS)**:\n"
            "   - Interoperable real-time mobile payment rails managed by NPCI, powering billions of low-cost P2P and P2M transactions.\n\n"
            "3. **Data Exchange Layer (DEPA & Account Aggregator)**:\n"
            "   - Consent-driven architecture empowering users to securely share financial, health (ABDM), and credential records (DigiLocker) without intermediary lock-in."
        ),
        "rejected": (
            "India Stack is an app you download on your phone to send money with Google Pay or PhonePe "
            "and show your Aadhaar card."
        ),
    },
]


def get_all_preference_samples() -> list[dict[str, Any]]:
    """Return complete consolidated list of preference samples."""
    return INDIC_PREFERENCES + STEM_AND_GLOBAL_PREFERENCES


def export_preference_curriculum(output_path: str | Path) -> int:
    """Export all preference pairs to a standardized JSONL file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    samples = get_all_preference_samples()
    with open(out, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(samples)
