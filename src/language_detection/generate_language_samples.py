"""generate_language_samples.py - Generates verified multilingual test dataset for Phase 5.

Includes:
- 12 English samples
- 12 Tamil samples
- 12 Hindi samples
- 12 Telugu samples
- 12 Malayalam samples
- 12 Kannada samples
- 7 OCR-derived samples from Phase 3/4
- 8 Edge cases (blank, whitespace, digits, symbols, extreme short)
- 6 Mixed-language samples
Total: 85 verified test samples with explicit ground truth.
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "evaluation" / "language_detection" / "language_samples.csv"

SAMPLES = [
    # ── ENGLISH (12 samples) ──
    {
        "sample_id": "en_01",
        "text": "The Reserve Bank of India kept the benchmark repo rate unchanged at 6.5 percent.",
        "expected_language": "English",
        "source": "financial_news",
        "notes": "Economic policy report"
    },
    {
        "sample_id": "en_02",
        "text": "Heavy rainfall warning issued for coastal districts following low pressure area over the Bay of Bengal.",
        "expected_language": "English",
        "source": "weather_report",
        "notes": "Meteorological bulletin"
    },
    {
        "sample_id": "en_03",
        "text": "The Indian Space Research Organisation announced the successful launch of its latest navigation satellite.",
        "expected_language": "English",
        "source": "science_news",
        "notes": "Space agency press release"
    },
    {
        "sample_id": "en_04",
        "text": "Ministry of Health confirms that national immunization coverage has reached record high levels this quarter.",
        "expected_language": "English",
        "source": "health_bulletin",
        "notes": "Public health announcement"
    },
    {
        "sample_id": "en_05",
        "text": "Global stock markets experienced mixed trading today ahead of central bank rate decisions.",
        "expected_language": "English",
        "source": "market_summary",
        "notes": "Financial market overview"
    },
    {
        "sample_id": "en_06",
        "text": "State government announces new infrastructure investment package for highway expansion and flood protection.",
        "expected_language": "English",
        "source": "governance_news",
        "notes": "Infrastructure policy release"
    },
    {
        "sample_id": "en_07",
        "text": "Scientists discover new biological indicators that improve early detection of soil moisture depletion.",
        "expected_language": "English",
        "source": "science_journal",
        "notes": "Agricultural science report"
    },
    {
        "sample_id": "en_08",
        "text": "Railway authorities have rescheduled multiple long-distance trains due to ongoing track modernization work.",
        "expected_language": "English",
        "source": "transport_bulletin",
        "notes": "Public transit advisory"
    },
    {
        "sample_id": "en_09",
        "text": "The National Disaster Management Authority conducted a simulated earthquake response drill across multiple cities.",
        "expected_language": "English",
        "source": "safety_news",
        "notes": "Disaster preparedness announcement"
    },
    {
        "sample_id": "en_10",
        "text": "Exports of agricultural commodities registered double digit growth in the first six months of the fiscal year.",
        "expected_language": "English",
        "source": "trade_report",
        "notes": "Trade statistics summary"
    },
    {
        "sample_id": "en_11",
        "text": "Educational board publishes finalized examination timetable for secondary school certifications.",
        "expected_language": "English",
        "source": "education_notice",
        "notes": "Academic schedule notice"
    },
    {
        "sample_id": "en_12",
        "text": "Renewable energy installations across southern states surpassed historical targets ahead of deadline.",
        "expected_language": "English",
        "source": "energy_report",
        "notes": "Clean energy development report"
    },

    # ── TAMIL (12 samples) ──
    {
        "sample_id": "ta_01",
        "text": "சென்னை வானிலை ஆய்வு மையம் இன்று பல மாவட்டங்களில் கனமழை எச்சரிக்கை விடுத்துள்ளது.",
        "expected_language": "Tamil",
        "source": "tamil_weather_news",
        "notes": "Weather forecast report"
    },
    {
        "sample_id": "ta_02",
        "text": "தமிழ்நாடு அரசு பள்ளிக்கல்வித்துறை புதிய பாடத்திட்டம் குறித்த முக்கிய அறிவிப்பை வெளியிட்டுள்ளது.",
        "expected_language": "Tamil",
        "source": "tamil_education_news",
        "notes": "Education department announcement"
    },
    {
        "sample_id": "ta_03",
        "text": "விவசாயிகளின் நலனுக்காக கூட்டுறவு வங்கிகள் மூலம் புதிய பயிர்க்கடன் திட்டம் தொடங்கப்பட்டுள்ளது.",
        "expected_language": "Tamil",
        "source": "tamil_agriculture_news",
        "notes": "Agricultural loan program announcement"
    },
    {
        "sample_id": "ta_04",
        "text": "சென்னையில் மெட்ரோ ரயில் இரண்டாம் கட்ட பணிகள் விரைவாக நடைபெற்று வருவதாக அதிகாரிகள் தெரிவித்துள்ளனர்.",
        "expected_language": "Tamil",
        "source": "tamil_infrastructure_news",
        "notes": "Metro transit development"
    },
    {
        "sample_id": "ta_05",
        "text": "மதுரை மீனாட்சி அம்மன் கோவில் திருவிழா முன்னேற்பாடுகள் குறித்து மாவட்ட ஆட்சியர் ஆலோசனை நடத்தினார்.",
        "expected_language": "Tamil",
        "source": "tamil_regional_news",
        "notes": "Cultural festival coordination"
    },
    {
        "sample_id": "ta_06",
        "text": "மின்சார வாரியம் பராமரிப்பு பணிகள் காரணமாக நாளை குறிப்பிட்ட பகுதிகளில் மின்தடை அறிவித்துள்ளது.",
        "expected_language": "Tamil",
        "source": "tamil_public_notice",
        "notes": "Electricity board maintenance notice"
    },
    {
        "sample_id": "ta_07",
        "text": "காவிரி நீர் பிடிப்பு பகுதிகளில் பெய்து வரும் தொடர் மழையால் அணைகளுக்கு நீர்வரத்து அதிகரித்துள்ளது.",
        "expected_language": "Tamil",
        "source": "tamil_water_report",
        "notes": "Reservoir water inflow report"
    },
    {
        "sample_id": "ta_08",
        "text": "தமிழகத்தில் தொழில் முதலீடுகளை ஈர்க்க புதிய தொழில் கொள்கை விரைவில் வெளியிடப்படும் என அமைச்சர் கூறினார்.",
        "expected_language": "Tamil",
        "source": "tamil_business_news",
        "notes": "Industrial policy announcement"
    },
    {
        "sample_id": "ta_09",
        "text": "அரசு மருத்துவமனைகளில் நவீன மருத்துவ உபகரணங்கள் நிறுவ கூடுதல் நிதி ஒதுக்கீடு செய்யப்பட்டுள்ளது.",
        "expected_language": "Tamil",
        "source": "tamil_health_news",
        "notes": "Public healthcare funding announcement"
    },
    {
        "sample_id": "ta_10",
        "text": "தீபாவளி பண்டிகையை முன்னிட்டு பொதுமக்கள் சொந்த ஊர் செல்ல சிறப்பு பேருந்துகள் இயக்கப்பட உள்ளன.",
        "expected_language": "Tamil",
        "source": "tamil_transport_news",
        "notes": "Festival special transit notice"
    },
    {
        "sample_id": "ta_11",
        "text": "மாநில அளவில் நடைபெற்ற அறிவியல் கண்காட்சியில் பள்ளி மாணவர்கள் தங்களின் புதிய படைப்புகளை காட்சிப்படுத்தினர்.",
        "expected_language": "Tamil",
        "source": "tamil_student_news",
        "notes": "State science exhibition report"
    },
    {
        "sample_id": "ta_12",
        "text": "தூத்துக்குடி துறைமுகம் மூலம் ஏற்றுமதி மற்றும் இறக்குமதி வர்த்தகம் புதிய சாதனை படைத்துள்ளதாக தெரிவிக்கப்பட்டது.",
        "expected_language": "Tamil",
        "source": "tamil_trade_news",
        "notes": "Port commercial trade bulletin"
    },

    # ── HINDI (12 samples) ──
    {
        "sample_id": "hi_01",
        "text": "भारतीय मौसम विज्ञान विभाग ने उत्तर भारत के कई राज्यों में भारी बारिश का अलर्ट जारी किया है।",
        "expected_language": "Hindi",
        "source": "hindi_weather_news",
        "notes": "National weather advisory"
    },
    {
        "sample_id": "hi_02",
        "text": "केंद्रीय मंत्रिमंडल ने किसानों के कल्याण के लिए नई आर्थिक सहायता योजना को मंजूरी दी है।",
        "expected_language": "Hindi",
        "source": "hindi_governance_news",
        "notes": "Cabinet agricultural policy"
    },
    {
        "sample_id": "hi_03",
        "text": "भारतीय रिजर्व बैंक ने मौद्रिक नीति समिति की बैठक के बाद ब्याज दरों को स्थिर रखने का निर्णय लिया।",
        "expected_language": "Hindi",
        "source": "hindi_financial_news",
        "notes": "Reserve Bank monetary policy"
    },
    {
        "sample_id": "hi_04",
        "text": "स्वास्थ्य मंत्रालय ने देशभर के सभी प्राथमिक स्वास्थ्य केंद्रों में डिजिटल सेवाएं शुरू करने की घोषणा की।",
        "expected_language": "Hindi",
        "source": "hindi_health_news",
        "notes": "Health ministry digital initiative"
    },
    {
        "sample_id": "hi_05",
        "text": "रेलवे प्रशासन ने आगामी त्योहारों के दौरान यात्रियों की सुविधा के लिए विशेष ट्रेनें चलाने का फैसला किया है।",
        "expected_language": "Hindi",
        "source": "hindi_railway_news",
        "notes": "Festival transit bulletin"
    },
    {
        "sample_id": "hi_06",
        "text": "भारतीय अंतरिक्ष अनुसंधान संगठन ने सौर मिशन के अंतर्गत एकत्र किए गए वैज्ञानिक आंकड़ों को साझा किया।",
        "expected_language": "Hindi",
        "source": "hindi_science_news",
        "notes": "Space research mission data"
    },
    {
        "sample_id": "hi_07",
        "text": "राज्य सरकार ने युवाओं को तकनीकी कौशल सिखाने के लिए नए प्रशिक्षण केंद्रों की स्थापना का प्रस्ताव रखा।",
        "expected_language": "Hindi",
        "source": "hindi_education_news",
        "notes": "Skill training initiative"
    },
    {
        "sample_id": "hi_08",
        "text": "सड़क परिवहन एवं राजमार्ग मंत्रालय ने राष्ट्रीय राजमार्गों के विस्तार के लिए नई परियोजनाएं स्वीकृत कीं।",
        "expected_language": "Hindi",
        "source": "hindi_infrastructure_news",
        "notes": "Highways development project"
    },
    {
        "sample_id": "hi_09",
        "text": "पर्यावरण मंत्रालय ने प्रदूषण नियंत्रण के लिए उद्योगों को कड़े दिशा-निर्देशों का पालन करने का निर्देश दिया।",
        "expected_language": "Hindi",
        "source": "hindi_environment_news",
        "notes": "Pollution control directives"
    },
    {
        "sample_id": "hi_10",
        "text": "शेयर बाजार में आज बैंकिंग और ऑटोमोबाइल कंपनियों के शेयरों में अच्छी बढ़त देखी गई।",
        "expected_language": "Hindi",
        "source": "hindi_market_news",
        "notes": "Stock market performance"
    },
    {
        "sample_id": "hi_11",
        "text": "ऊर्जा क्षेत्र में नवीकरणीय स्रोतों के योगदान में निरंतर वृद्धि दर्ज की जा रही है।",
        "expected_language": "Hindi",
        "source": "hindi_energy_news",
        "notes": "Renewable energy sector summary"
    },
    {
        "sample_id": "hi_12",
        "text": "खेल मंत्रालय ने आगामी अंतरराष्ट्रीय प्रतियोगिताओं की तैयारी कर रहे खिलाड़ियों के लिए विशेष वित्तीय सहायता की घोषणा की।",
        "expected_language": "Hindi",
        "source": "hindi_sports_news",
        "notes": "Athletic funding announcement"
    },

    # ── TELUGU (12 samples) ──
    {
        "sample_id": "te_01",
        "text": "బంగాళాఖాతంలో ఏర్పడిన అల్పపీడనం కారణంగా కోస్తాంధ్ర జిల్లాల్లో భారీ వర్షాలు కురిసే అవకాశం ఉంది.",
        "expected_language": "Telugu",
        "source": "telugu_weather_news",
        "notes": "Coastal rainfall warning"
    },
    {
        "sample_id": "te_02",
        "text": "రైతుల సంక్షేమం కోసం ప్రభుత్వం కొత్త సాగునీటి ప్రాజెక్టుల నిర్మాణాన్ని వేగవంతం చేస్తోంది.",
        "expected_language": "Telugu",
        "source": "telugu_agriculture_news",
        "notes": "Irrigation project announcement"
    },
    {
        "sample_id": "te_03",
        "text": "హైదరాబాద్ నగరంలో మెట్రో రైలు రెండో దశ విస్తరణ పనులకు సన్నాహాలు ప్రారంభమయ్యాయి.",
        "expected_language": "Telugu",
        "source": "telugu_infrastructure_news",
        "notes": "Hyderabad metro phase 2"
    },
    {
        "sample_id": "te_04",
        "text": "పాఠశాలల్లో మౌలిక వసతుల కల్పనకు విద్యాశాఖ ప్రత్యేక నిధులు కేటాయించినట్లు అధికారులు తెలిపారు.",
        "expected_language": "Telugu",
        "source": "telugu_education_news",
        "notes": "School infrastructure allocation"
    },
    {
        "sample_id": "te_05",
        "text": "వైద్యారోగ్య రంగంలో నూతన సంస్కరణలు అమలు చేసి గ్రామీణ ప్రాంతాలకు మెరుగైన వైద్యం అందిస్తామని మంత్రి పేర్కొన్నారు.",
        "expected_language": "Telugu",
        "source": "telugu_health_news",
        "notes": "Rural healthcare development"
    },
    {
        "sample_id": "te_06",
        "text": "పండుగల సీజన్ సందర్భంగా ప్రయాణికుల రద్దీని దృష్టిలో ఉంచుకుని అదనపు ఆర్టీసీ బస్సులను నడుపుతున్నారు.",
        "expected_language": "Telugu",
        "source": "telugu_transport_news",
        "notes": "Transit schedule advisory"
    },
    {
        "sample_id": "te_07",
        "text": "రాష్ట్రంలో పారిశ్రామిక పెట్టుబడులను ఆకర్షించేందుకు ప్రభుత్వం ప్రత్యేక రాయితీలను ప్రకటించింది.",
        "expected_language": "Telugu",
        "source": "telugu_business_news",
        "notes": "Industrial investment policy"
    },
    {
        "sample_id": "te_08",
        "text": "శ్రీశైలం మరియు నాగార్జునసాగర్ జలాశయాలకు ఎగువ ప్రాంతాల నుంచి వరద నీరు భారీగా వచ్చి చేరుతోంది.",
        "expected_language": "Telugu",
        "source": "telugu_water_news",
        "notes": "Reservoir inflow report"
    },
    {
        "sample_id": "te_09",
        "text": "యువతకు ఉపాధి అవకాశాలు కల్పించేందుకు నైపుణ్యాభివృద్ధి సంస్థ ఉచిత శిక్షణ తరగతులను నిర్వహిస్తోంది.",
        "expected_language": "Telugu",
        "source": "telugu_employment_news",
        "notes": "Skill development program"
    },
    {
        "sample_id": "te_10",
        "text": "పర్యావరణ పరిరక్షణ కోసం నగరాల్లో మొక్కలు నాటే కార్యక్రమాన్ని విస్తృతంగా చేపట్టాలని అధికారులు పిలుపునిచ్చారు.",
        "expected_language": "Telugu",
        "source": "telugu_environment_news",
        "notes": "Urban plantation initiative"
    },
    {
        "sample_id": "te_11",
        "text": "శాస్త్ర సాంకేతిక రంగాలలో పరిశోధనలను ప్రోత్సహించేందుకు విశ్వవిద్యాలయాల్లో ప్రత్యేక ల్యాబ్‌లు ఏర్పాటు చేస్తున్నారు.",
        "expected_language": "Telugu",
        "source": "telugu_science_news",
        "notes": "University laboratory funding"
    },
    {
        "sample_id": "te_12",
        "text": "జాతీయ స్థాయి క్రీడా పోటీలలో తెలుగు రాష్ట్రాల క్రీడాకారులు అద్భుత ప్రతిభ కనబరిచి పతకాలు సాధించారు.",
        "expected_language": "Telugu",
        "source": "telugu_sports_news",
        "notes": "Sports medal victory report"
    },

    # ── MALAYALAM (12 samples) ──
    {
        "sample_id": "ml_01",
        "text": "സംസ്ഥാനത്ത് വിവിധ ജില്ലകളിൽ അടുത്ത ദിവസങ്ങളിൽ ശക്തമായ മഴയ്ക്ക് സാധ്യതയുണ്ടെന്ന് കാലാവസ്ഥാ നിരീക്ഷണ കേന്ദ്രം അറിയിച്ചു.",
        "expected_language": "Malayalam",
        "source": "malayalam_weather_news",
        "notes": "Heavy rain forecast alert"
    },
    {
        "sample_id": "ml_02",
        "text": "പൊതുവിദ്യാഭ്യാസ സംരക്ഷണ യജ്ഞത്തിന്റെ ഭാഗമായി സ്കൂളുകളിൽ ഡിജിറ്റൽ ക്ലാസ് മുറികൾ പൂർണ്ണ സജ്ജമായി.",
        "expected_language": "Malayalam",
        "source": "malayalam_education_news",
        "notes": "Digital school classrooms"
    },
    {
        "sample_id": "ml_03",
        "text": "കൃഷിവകുപ്പ് കർഷകർക്കായി പുതിയ വിള ഇൻഷുറൻസ് പദ്ധതി പ്രഖ്യാപിച്ചു.",
        "expected_language": "Malayalam",
        "source": "malayalam_agriculture_news",
        "notes": "Crop insurance program"
    },
    {
        "sample_id": "ml_04",
        "text": "കൊച്ചി മെട്രോയുടെ പുതിയ പാതയുടെ നിർമ്മാണ പ്രവർത്തനങ്ങൾ വേഗത്തിൽ പുരോഗമിക്കുന്നതായി അധികൃതർ വ്യക്തമാക്കി.",
        "expected_language": "Malayalam",
        "source": "malayalam_infrastructure_news",
        "notes": "Kochi metro expansion"
    },
    {
        "sample_id": "ml_05",
        "text": "ആരോഗ്യവകുപ്പ് ആശുപത്രികളിൽ അത്യാധുനിക ചികിത്സാ സൗകര്യങ്ങൾ ഒരുക്കുന്നതിനായി കൂടുതൽ ഫണ്ട് അനുവദിച്ചു.",
        "expected_language": "Malayalam",
        "source": "malayalam_health_news",
        "notes": "Healthcare infrastructure fund"
    },
    {
        "sample_id": "ml_06",
        "text": "ഓണക്കാലത്തെ തിരക്ക് നിയന്ത്രിക്കുന്നതിനായി പ്രത്യേക ട്രെയിനുകളും ബസുകളും സർവീസ് നടത്തും.",
        "expected_language": "Malayalam",
        "source": "malayalam_transport_news",
        "notes": "Festival transport services"
    },
    {
        "sample_id": "ml_07",
        "text": "സംസ്ഥാനത്തെ ഐടി മേഖലയിൽ കൂടുതൽ തൊഴിലവസരങ്ങൾ സൃഷ്ടിക്കുന്നതിനായി പുതിയ പാർക്കുകൾ ആരംഭിക്കും.",
        "expected_language": "Malayalam",
        "source": "malayalam_it_news",
        "notes": "IT park employment expansion"
    },
    {
        "sample_id": "ml_08",
        "text": "വനമേഖലയോട് ചേർന്നുള്ള ജനവാസ കേന്ദ്രങ്ങളിൽ വന്യജീവി ശല്യം തടയാൻ പ്രതിരോധ സംവിധാനങ്ങൾ ശക്തമാക്കും.",
        "expected_language": "Malayalam",
        "source": "malayalam_forest_news",
        "notes": "Wildlife protection measures"
    },
    {
        "sample_id": "ml_09",
        "text": "വിനോദസഞ്ചാര മേഖലയുടെ വികസനത്തിനായി പുതിയ പരിസ്ഥിതി സൗഹൃദ പദ്ധതികൾ നടപ്പിലാക്കുന്നു.",
        "expected_language": "Malayalam",
        "source": "malayalam_tourism_news",
        "notes": "Eco-tourism initiatives"
    },
    {
        "sample_id": "ml_10",
        "text": "തീരദേശ സംരക്ഷണത്തിനായി പുതിയ പുലിമുട്ടുകളുടെ നിർമ്മാണം ഉടൻ ആരംഭിക്കുമെന്ന് മന്ത്രി അറിയിച്ചു.",
        "expected_language": "Malayalam",
        "source": "malayalam_coastal_news",
        "notes": "Coastal embankment protection"
    },
    {
        "sample_id": "ml_11",
        "text": "പരമ്പരാഗത മത്സ്യത്തൊഴിലാളികൾക്ക് സുരക്ഷാ ഉപകരണങ്ങളും ധനസഹായവും ലഭ്യമാക്കുമെന്ന് സർക്കാർ വ്യക്തമാക്കി.",
        "expected_language": "Malayalam",
        "source": "malayalam_fisheries_news",
        "notes": "Fisheries welfare scheme"
    },
    {
        "sample_id": "ml_12",
        "text": "ശാസ്ത്ര സാങ്കേതിക രംഗത്ത് നൂതന ആശയങ്ങൾ അവതരിപ്പിച്ച വിദ്യാർത്ഥികൾക്ക് പുരസ്കാരങ്ങൾ വിതരണം ചെയ്തു.",
        "expected_language": "Malayalam",
        "source": "malayalam_science_news",
        "notes": "Student innovation awards"
    },

    # ── KANNADA (12 samples) ──
    {
        "sample_id": "kn_01",
        "text": "ಕರಾವಳಿ ಮತ್ತು ಮಲೆನಾಡು ಭಾಗಗಳಲ್ಲಿ ಮುಂದಿನ ಎರಡು ದಿನಗಳ ಕಾಲ ಭಾರಿ ಮಳೆಯಾಗುವ ಮುನ್ಸೂಚನೆಯನ್ನು ಹವಾಮಾನ ಇಲಾಖೆ ನೀಡಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_weather_news",
        "notes": "Coastal weather forecast"
    },
    {
        "sample_id": "kn_02",
        "text": "ರೈತರ ಆರ್ಥಿಕ ಪ್ರಗತಿಗಾಗಿ ಕೃಷಿ ಇಲಾಖೆಯು ಹೊಸ ಸಬ್ಸಿಡಿ ಯೋಜನೆಯನ್ನು ಜಾರಿಗೆ ತಂದಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_agriculture_news",
        "notes": "Agricultural subsidy program"
    },
    {
        "sample_id": "kn_03",
        "text": "ಬೆಂಗಳೂರು ನಮ್ಮ ಮೆಟ್ರೋ ಮೂರನೇ ಹಂತದ ಕಾಮಗಾರಿಗೆ ಕೇಂದ್ರ ಸರ್ಕಾರ ಅನುಮೋದನೆ ನೀಡಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_infrastructure_news",
        "notes": "Bengaluru Namma Metro phase 3"
    },
    {
        "sample_id": "kn_04",
        "text": "ಶಿಕ್ಷಣ ಇಲಾಖೆಯು ಸರ್ಕಾರಿ ಶಾಲೆಗಳ ಆಧುನೀಕರಣಕ್ಕಾಗಿ ವಿಶೇಷ ಅನುದಾನ ಬಿಡುಗಡೆ ಮಾಡಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_education_news",
        "notes": "Public school modernization"
    },
    {
        "sample_id": "kn_05",
        "text": "ಆರೋಗ್ಯ ಇಲಾಖೆಯು ಗ್ರಾಮೀಣ ಪ್ರದೇಶಗಳಲ್ಲಿ ಉಚಿತ ಆರೋಗ್ಯ ತಪಾಸಣಾ ಶಿಬಿರಗಳನ್ನು ಆಯೋಜಿಸಲು ನಿರ್ಧರಿಸಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_health_news",
        "notes": "Rural health camp initiative"
    },
    {
        "sample_id": "kn_06",
        "text": "ಹಬ್ಬದ ಪ್ರಯುಕ್ತ ಸಾರ್ವಜನಿಕರ ಅನುಕೂಲಕ್ಕಾಗಿ ರಾಜ್ಯ ರಸ್ತೆ ಸಾರಿಗೆ ಸಂಸ್ಥೆಯು ಹೆಚ್ಚುವರಿ ಬಸ್‌ಗಳನ್ನು ಓಡಿಸಲಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_transport_news",
        "notes": "State transport festival service"
    },
    {
        "sample_id": "kn_07",
        "text": "ರಾಜ್ಯದಲ್ಲಿ ಕೈಗಾರಿಕಾ ಹೂಡಿಕೆಗಳನ್ನು ಹೆಚ್ಚಿಸಲು ಹೊಸ ನೀತಿಯನ್ನು ಶೀಘ್ರದಲ್ಲೇ ಪ್ರಕಟಿಸಲಾಗುವುದು ಎಂದು ಕೈಗಾರಿಕಾ ಸಚಿವರು ಹೇಳಿದರು.",
        "expected_language": "Kannada",
        "source": "kannada_business_news",
        "notes": "Industrial investment policy"
    },
    {
        "sample_id": "kn_08",
        "text": "ಕಾವೇರಿ ಜಲಾನಯನ ಪ್ರದೇಶದಲ್ಲಿ ಮಳೆ ಚುರುಕುಗೊಂಡಿದ್ದು, ಜಲಾಶಯಗಳಿಗೆ ನೀರಿನ ಒಳಹರಿವು ಗಣನೀಯವಾಗಿ ಹೆಚ್ಚಾಗಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_water_news",
        "notes": "Cauvery basin reservoir inflow"
    },
    {
        "sample_id": "kn_09",
        "text": "ಯುವಜನರಿಗೆ ಉದ್ಯೋಗಾವಕಾಶ ಕಲ್ಪಿಸುವ ನಿಟ್ಟಿನಲ್ಲಿ ಬೃಹತ್ ಉದ್ಯೋಗ ಮೇಳವನ್ನು ಆಯೋಜಿಸಲಾಗಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_employment_news",
        "notes": "Youth job fair announcement"
    },
    {
        "sample_id": "kn_10",
        "text": "ಪರಿಸರ ಸಂರಕ್ಷಣೆಗಾಗಿ ಪ್ಲಾಸ್ಟಿಕ್ ಬಳಕೆಯನ್ನು ಕಟ್ಟುನಿಟ್ಟಾಗಿ ನಿಷೇಧಿಸಲು ನಗರ ಪಾಲಿಕೆ ಕ್ರಮ ಕೈಗೊಂಡಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_environment_news",
        "notes": "Plastic ban enforcement"
    },
    {
        "sample_id": "kn_11",
        "text": "ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಕ್ಷೇತ್ರದಲ್ಲಿ ಸಂಶೋಧನೆ ನಡೆಸುತ್ತಿರುವ ಯುವ ವಿಜ್ಞಾನಿಗಳಿಗೆ ರಾಜ್ಯ ಪ್ರಶಸ್ತಿ ಪ್ರದಾನ ಮಾಡಲಾಯಿತು.",
        "expected_language": "Kannada",
        "source": "kannada_science_news",
        "notes": "Young scientist awards"
    },
    {
        "sample_id": "kn_12",
        "text": "ಐತಿಹಾಸಿಕ ಪ್ರವಾಸಿ ತಾಣಗಳ ರಕ್ಷಣೆ ಮತ್ತು ನವೀಕರಣಕ್ಕಾಗಿ ಪ್ರವಾಸೋದ್ಯಮ ಇಲಾಖೆ ಸಮಗ್ರ ಯೋಜನೆ ರೂಪಿಸಿದೆ.",
        "expected_language": "Kannada",
        "source": "kannada_tourism_news",
        "notes": "Heritage tourism conservation"
    },

    # ── OCR-DERIVED SAMPLES (from Phase 3/4 outputs) (7 samples) ──
    {
        "sample_id": "ocr_en_clear",
        "text": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "expected_language": "English",
        "source": "phase3_ocr_output",
        "notes": "OCR clean text from english_clear.png"
    },
    {
        "sample_id": "ocr_en_noisy",
        "text": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "expected_language": "English",
        "source": "phase3_ocr_output",
        "notes": "OCR clean text from english_noisy.png"
    },
    {
        "sample_id": "ocr_en_skewed",
        "text": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "expected_language": "English",
        "source": "phase3_ocr_output",
        "notes": "OCR clean text from english_skewed.png"
    },
    {
        "sample_id": "ocr_en_low_contrast",
        "text": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "expected_language": "English",
        "source": "phase3_ocr_output",
        "notes": "OCR clean text from english_low_contrast.png"
    },
    {
        "sample_id": "ocr_ta_sample",
        "text": "சென்னை வானிலை ஆய்வு மையம்‌ இன்று மழை எச்சரிக்கை விடுத்துள்ளது.",
        "expected_language": "Tamil",
        "source": "phase3_ocr_output",
        "notes": "OCR clean text from tamil_news_sample.png (contains U+200C)"
    },
    {
        "sample_id": "ocr_blank",
        "text": "",
        "expected_language": "N/A",
        "source": "phase3_ocr_output",
        "notes": "OCR output from blank.png (empty string)"
    },
    {
        "sample_id": "ocr_tiny",
        "text": "",
        "expected_language": "N/A",
        "source": "phase3_ocr_output",
        "notes": "OCR output from tiny.png (empty string)"
    },

    # ── EDGE CASES (8 samples) ──
    {
        "sample_id": "edge_empty",
        "text": "",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Completely empty input"
    },
    {
        "sample_id": "edge_whitespace_only",
        "text": "   \n\t   \n  ",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Whitespace only characters"
    },
    {
        "sample_id": "edge_single_char_en",
        "text": "a",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Single English character"
    },
    {
        "sample_id": "edge_single_char_ta",
        "text": "த",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Single Tamil character"
    },
    {
        "sample_id": "edge_numeric_only",
        "text": "1234567890 9876543210 2026",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Numbers only"
    },
    {
        "sample_id": "edge_punctuation_only",
        "text": "... ??? !!! --- @@@ ### $$$ %%%",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Punctuation and symbols only"
    },
    {
        "sample_id": "edge_short_word_en",
        "text": "hi",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Two letter English word below threshold"
    },
    {
        "sample_id": "edge_short_word_ta",
        "text": "வா",
        "expected_language": "N/A",
        "source": "synthetic_edge_case",
        "notes": "Short Tamil word below linguistic character threshold"
    },

    # ── MIXED-LANGUAGE TEXT (6 samples) ──
    {
        "sample_id": "mixed_ta_en_01",
        "text": "Chennai-யில் heavy rain எச்சரிக்கை விடுக்கப்பட்டுள்ளது என வானிலை மையம் தெரிவித்தது.",
        "expected_language": "Tamil",
        "source": "synthetic_code_mixed",
        "notes": "Tamil dominant with English terms (Chennai, heavy rain)"
    },
    {
        "sample_id": "mixed_ta_en_02",
        "text": "Tamil Nadu government announces new IT park in Coimbatore with 5000 job opportunities.",
        "expected_language": "English",
        "source": "synthetic_code_mixed",
        "notes": "English dominant with Tamil proper nouns (Tamil Nadu, Coimbatore)"
    },
    {
        "sample_id": "mixed_hi_en_01",
        "text": "Delhi Metro ने passengers के लिए new smart card system शुरू किया है।",
        "expected_language": "Hindi",
        "source": "synthetic_code_mixed",
        "notes": "Hindi dominant with English words (passengers, smart card system)"
    },
    {
        "sample_id": "mixed_hi_en_02",
        "text": "The Prime Minister inaugurated the new AIIMS campus in Bilaspur today.",
        "expected_language": "English",
        "source": "synthetic_code_mixed",
        "notes": "English with Hindi acronym/proper noun (AIIMS, Bilaspur)"
    },
    {
        "sample_id": "mixed_te_en_01",
        "text": "Hyderabad Cyber Crime Police warning issued regarding online investment scams.",
        "expected_language": "English",
        "source": "synthetic_code_mixed",
        "notes": "English text with Telugu regional origin"
    },
    {
        "sample_id": "mixed_te_en_02",
        "text": "విజయవాడలో కొత్త software technology park ఏర్పాటుకు ప్రభుత్వం ఆమోదం తెలిపింది.",
        "expected_language": "Telugu",
        "source": "synthetic_code_mixed",
        "notes": "Telugu dominant with English loan words (software technology park)"
    },
]


def generate_samples_csv():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "text", "expected_language", "source", "notes"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(SAMPLES)
    print(f"Generated {len(SAMPLES)} samples in {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_samples_csv()
