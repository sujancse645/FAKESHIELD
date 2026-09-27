"""generate_translation_samples.py - Generates verified multilingual evaluation dataset for Phase 6.

Includes 11 target languages (6 samples each) + OCR-derived samples + edge cases.
Total: ~75 verified test samples with independent, human-verified English reference translations.
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "evaluation" / "translation" / "translation_samples.csv"

SAMPLES = [
    # ── 1. ENGLISH (Pivot - should be skipped) (6 samples) ──
    {
        "sample_id": "trans_en_01",
        "source_text": "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent on 15 September 2026.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "The Reserve Bank of India kept the benchmark interest rate unchanged at 6.5 percent on 15 September 2026.",
        "category": "financial_news",
        "source_type": "clean_text",
        "notes": "English pivot sample with date and percentage"
    },
    {
        "sample_id": "trans_en_02",
        "source_text": "Health authorities did not find any evidence of contamination in the public water supply.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "Health authorities did not find any evidence of contamination in the public water supply.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "English pivot sample with explicit negation"
    },
    {
        "sample_id": "trans_en_03",
        "source_text": "ISRO successfully placed 3 satellites into polar orbit from Sriharikota.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "ISRO successfully placed 3 satellites into polar orbit from Sriharikota.",
        "category": "science_news",
        "source_type": "clean_text",
        "notes": "English pivot sample with numbers and named entities"
    },
    {
        "sample_id": "trans_en_04",
        "source_text": "The government will allocate 500 crore rupees for flood relief operations.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "The government will allocate 500 crore rupees for flood relief operations.",
        "category": "governance_news",
        "source_type": "clean_text",
        "notes": "English pivot sample with currency and financial quantity"
    },
    {
        "sample_id": "trans_en_05",
        "source_text": "Chennai Metro Rail phase 2 construction is progressing according to schedule.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "Chennai Metro Rail phase 2 construction is progressing according to schedule.",
        "category": "infrastructure_news",
        "source_type": "clean_text",
        "notes": "English pivot sample with named entity and number"
    },
    {
        "sample_id": "trans_en_06",
        "source_text": "National Highway Authority confirmed that toll rates will not increase this month.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "National Highway Authority confirmed that toll rates will not increase this month.",
        "category": "transport_news",
        "source_type": "clean_text",
        "notes": "English pivot sample with negation"
    },

    # ── 2. TAMIL (6 samples) ──
    {
        "sample_id": "trans_ta_01",
        "source_text": "சென்னை வானிலை ஆய்வு மையம் இன்று பல மாவட்டங்களில் கனமழை எச்சரிக்கை விடுத்துள்ளது.",
        "source_language": "ta",
        "source_language_name": "Tamil",
        "reference_english": "The Chennai Meteorological Center has issued a heavy rain warning for many districts today.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Named entity (Chennai Meteorological Center) and weather warning"
    },
    {
        "sample_id": "trans_ta_02",
        "source_text": "விவசாயிகளுக்கு 50000 ரூபாய் வட்டியில்லா பயிர்க்கடன் வழங்கப்படும் என முதலமைச்சர் அறிவித்தார்.",
        "source_language": "ta",
        "source_language_name": "Tamil",
        "reference_english": "The Chief Minister announced that interest-free crop loans of 50000 rupees will be provided to farmers.",
        "category": "agriculture_news",
        "source_type": "clean_text",
        "notes": "Numeric quantity (50000) and government title"
    },
    {
        "sample_id": "trans_ta_03",
        "source_text": "அரசு மருத்துவமனைகளில் மருந்து தட்டுப்பாடு இல்லை என்று சுகாதாரத்துறை அமைச்சர் உறுதி செய்தார்.",
        "source_language": "ta",
        "source_language_name": "Tamil",
        "reference_english": "The Health Minister confirmed that there is no shortage of medicines in government hospitals.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "Explicit negation (இல்லை - no shortage)"
    },
    {
        "sample_id": "trans_ta_04",
        "source_text": "15 அக்டோபர் 2026 அன்று மதுரை மற்றும் திருச்சியில் புதிய விமான சேவைகள் தொடங்கப்படும்.",
        "source_language": "ta",
        "source_language_name": "Tamil",
        "reference_english": "New air services will begin in Madurai and Trichy on 15 October 2026.",
        "category": "transport_news",
        "source_type": "clean_text",
        "notes": "Exact date (15 October 2026) and place names (Madurai, Trichy)"
    },
    {
        "sample_id": "trans_ta_05",
        "source_text": "தமிழகத்தில் 12 புதிய தொழில் பூங்காக்கள் அமைக்க 2500 கோடி ரூபாய் முதலீடு செய்யப்படுகிறது.",
        "source_language": "ta",
        "source_language_name": "Tamil",
        "reference_english": "An investment of 2500 crore rupees is being made to establish 12 new industrial parks in Tamil Nadu.",
        "category": "business_news",
        "source_type": "clean_text",
        "notes": "Multiple numbers (12, 2500) and place name (Tamil Nadu)"
    },
    {
        "sample_id": "trans_ta_06",
        "source_text": "பள்ளி மாணவர்களுக்கு இலவச சீருடைகள் மற்றும் பாடப்புத்தகங்கள் வழங்கப்பட்டன.",
        "source_language": "ta",
        "source_language_name": "Tamil",
        "reference_english": "Free uniforms and textbooks were distributed to school students.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "General education welfare news"
    },

    # ── 3. HINDI (6 samples) ──
    {
        "sample_id": "trans_hi_01",
        "source_text": "भारतीय रिजर्व बैंक ने आज मौद्रिक नीति में कोई बदलाव नहीं करने का फैसला किया।",
        "source_language": "hi",
        "source_language_name": "Hindi",
        "reference_english": "The Reserve Bank of India decided not to make any changes in monetary policy today.",
        "category": "financial_news",
        "source_type": "clean_text",
        "notes": "Named entity (Reserve Bank of India) and negation (नहीं)"
    },
    {
        "sample_id": "trans_hi_02",
        "source_text": "मौसम विभाग ने उत्तर भारत के 5 राज्यों में अगले 48 घंटों के लिए भारी बारिश का अलर्ट जारी किया।",
        "source_language": "hi",
        "source_language_name": "Hindi",
        "reference_english": "The Meteorological Department issued a heavy rain alert for 5 states in North India for the next 48 hours.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Numbers (5, 48) and geographic region"
    },
    {
        "sample_id": "trans_hi_03",
        "source_text": "सरकार ने 10000 किलोमीटर नए राष्ट्रीय राजमार्ग बनाने के लिए 80000 करोड़ रुपये मंजूर किए।",
        "source_language": "hi",
        "source_language_name": "Hindi",
        "reference_english": "The government approved 80000 crore rupees to construct 10000 kilometers of new national highways.",
        "category": "infrastructure_news",
        "source_type": "clean_text",
        "notes": "Multiple numeric quantities (10000, 80000)"
    },
    {
        "sample_id": "trans_hi_04",
        "source_text": "स्वास्थ्य मंत्रालय ने स्पष्ट किया कि टीकाकरण के बाद कोई भी गंभीर प्रतिकूल प्रभाव नहीं देखा गया।",
        "source_language": "hi",
        "source_language_name": "Hindi",
        "reference_english": "The Health Ministry clarified that no serious adverse effects were observed after vaccination.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "Explicit negation (नहीं देखा गया)"
    },
    {
        "sample_id": "trans_hi_05",
        "source_text": "इसरो 20 नवंबर 2026 को श्रीहरिकोटा से नया संचार उपग्रह प्रक्षेपित करेगा।",
        "source_language": "hi",
        "source_language_name": "Hindi",
        "reference_english": "ISRO will launch a new communication satellite from Sriharikota on 20 November 2026.",
        "category": "science_news",
        "source_type": "clean_text",
        "notes": "Date (20 November 2026) and named entity (ISRO, Sriharikota)"
    },
    {
        "sample_id": "trans_hi_06",
        "source_text": "कृषि विश्वविद्यालय ने गेहूं की दो नई उच्च उपज वाली किस्मों का विकास किया।",
        "source_language": "hi",
        "source_language_name": "Hindi",
        "reference_english": "The Agricultural University developed two new high-yielding varieties of wheat.",
        "category": "agriculture_news",
        "source_type": "clean_text",
        "notes": "Agricultural science report"
    },

    # ── 4. TELUGU (6 samples) ──
    {
        "sample_id": "trans_te_01",
        "source_text": "విశాఖపట్నం తీరంలో తీవ్ర తుపాను కారణంగా అధికారులు హై అలర్ట్ ప్రకటించారు.",
        "source_language": "te",
        "source_language_name": "Telugu",
        "reference_english": "Authorities declared a high alert due to a severe cyclone off the Visakhapatnam coast.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Place name (Visakhapatnam) and weather alert"
    },
    {
        "sample_id": "trans_te_02",
        "source_text": "రైతులకు 24 గంటల ఉచిత విద్యుత్ సరఫరా కొనసాగుతుందని ప్రభుత్వం స్పష్టం చేసింది.",
        "source_language": "te",
        "source_language_name": "Telugu",
        "reference_english": "The government clarified that 24-hour free electricity supply to farmers will continue.",
        "category": "energy_news",
        "source_type": "clean_text",
        "notes": "Number (24) and government policy statement"
    },
    {
        "sample_id": "trans_te_03",
        "source_text": "హైదరాబాద్ నగరంలో నీటి కొరత లేదని జలమండలి అధికారులు తెలిపారు.",
        "source_language": "te",
        "source_language_name": "Telugu",
        "reference_english": "Water board officials stated that there is no water shortage in Hyderabad city.",
        "category": "utilities_news",
        "source_type": "clean_text",
        "notes": "Place name (Hyderabad) and negation (లేదని - no shortage)"
    },
    {
        "sample_id": "trans_te_04",
        "source_text": "ఆంధ్రప్రదేశ్ ప్రభుత్వం 12000 ఉపాధ్యాయ పోస్టుల భర్తీకి నోటిఫికేషన్ విడుదల చేసింది.",
        "source_language": "te",
        "source_language_name": "Telugu",
        "reference_english": "The Andhra Pradesh government issued a notification to fill 12000 teacher posts.",
        "category": "employment_news",
        "source_type": "clean_text",
        "notes": "Place name (Andhra Pradesh) and quantity (12000)"
    },
    {
        "sample_id": "trans_te_05",
        "source_text": "శ్రీశైలం ప్రాజెక్టుకు ఎగువ ప్రాంతాల నుంచి 50000 క్యూసెక్కుల వరద నీరు వస్తోంది.",
        "source_language": "te",
        "source_language_name": "Telugu",
        "reference_english": "Flood water of 50000 cusecs is arriving at the Srisailam project from upstream areas.",
        "category": "water_news",
        "source_type": "clean_text",
        "notes": "Proper noun (Srisailam) and numeric measure (50000)"
    },
    {
        "sample_id": "trans_te_06",
        "source_text": "విద్యా ప్రమాణాలు మెరుగుపరచడానికి ప్రభుత్వ పాఠశాలల్లో డిజిటల్ తరగతి గదులు ఏర్పాటు చేస్తున్నారు.",
        "source_language": "te",
        "source_language_name": "Telugu",
        "reference_english": "Digital classrooms are being established in government schools to improve educational standards.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "Education development statement"
    },

    # ── 5. MALAYALAM (6 samples) ──
    {
        "sample_id": "trans_ml_01",
        "source_text": "ഇടുക്കി ജില്ലയിൽ അതിശക്തമായ മഴ പെയ്യാൻ സാധ്യതയുള്ളതിനാൽ റെഡ് അലർട്ട് പ്രഖ്യാപിച്ചു.",
        "source_language": "ml",
        "source_language_name": "Malayalam",
        "reference_english": "A red alert was declared in Idukki district due to the likelihood of very heavy rain.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Place name (Idukki) and weather alert"
    },
    {
        "sample_id": "trans_ml_02",
        "source_text": "സംസ്ഥാനത്ത് ഭക്ഷ്യധാന്യങ്ങളുടെ വില വർധിച്ചിട്ടില്ലെന്ന് മന്ത്രി നിയമസഭയിൽ അറിയിച്ചു.",
        "source_language": "ml",
        "source_language_name": "Malayalam",
        "reference_english": "The Minister informed the Assembly that food grain prices have not increased in the state.",
        "category": "governance_news",
        "source_type": "clean_text",
        "notes": "Negation (വർധിച്ചിട്ടില്ലെന്ന് - have not increased)"
    },
    {
        "sample_id": "trans_ml_03",
        "source_text": "കൊച്ചി മെട്രോയുടെ പുതിയ പാത 2026 ഡിസംബറിൽ കമ്മീഷൻ ചെയ്യുമെന്ന് കെഎംആർഎൽ വ്യക്തമാക്കി.",
        "source_language": "ml",
        "source_language_name": "Malayalam",
        "reference_english": "KMRL stated that the new Kochi Metro line will be commissioned in December 2026.",
        "category": "transport_news",
        "source_type": "clean_text",
        "notes": "Date (December 2026) and proper nouns (Kochi Metro, KMRL)"
    },
    {
        "sample_id": "trans_ml_04",
        "source_text": "തീരദേശ സംരക്ഷണത്തിനായി സർക്കാർ 1500 കോടി രൂപയുടെ പദ്ധതിക്ക് അനുമതി നൽകി.",
        "source_language": "ml",
        "source_language_name": "Malayalam",
        "reference_english": "The government approved a project of 1500 crore rupees for coastal protection.",
        "category": "coastal_news",
        "source_type": "clean_text",
        "notes": "Financial quantity (1500 crore)"
    },
    {
        "sample_id": "trans_ml_05",
        "source_text": "ആശുപത്രികളിൽ ജീവൻരക്ഷാ മരുന്നുകൾ ആവശ്യത്തിന് ലഭ്യമാണെന്ന് ആരോഗ്യവകുപ്പ് അറിയിച്ചു.",
        "source_language": "ml",
        "source_language_name": "Malayalam",
        "reference_english": "The Health Department informed that life-saving medicines are adequately available in hospitals.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "Healthcare administration statement"
    },
    {
        "sample_id": "trans_ml_06",
        "source_text": "സംസ്ഥാനത്ത് 500 പുതിയ സ്കൂൾ ബസുകൾ വാങ്ങാൻ ഗതാഗത വകുപ്പ് തീരുമാനിച്ചു.",
        "source_language": "ml",
        "source_language_name": "Malayalam",
        "reference_english": "The Transport Department decided to purchase 500 new school buses in the state.",
        "category": "transport_news",
        "source_type": "clean_text",
        "notes": "Quantity (500) and department name"
    },

    # ── 6. KANNADA (6 samples) ──
    {
        "sample_id": "trans_kn_01",
        "source_text": "ಬೆಂಗಳೂರು ನಗರದಲ್ಲಿ ಕುಡಿಯುವ ನೀರಿನ ಸಮಸ್ಯೆ ಇಲ್ಲ ಎಂದು ಜಲಮಂಡಳಿ ಸ್ಪಷ್ಟಪಡಿಸಿದೆ.",
        "source_language": "kn",
        "source_language_name": "Kannada",
        "reference_english": "The water board clarified that there is no drinking water problem in Bengaluru city.",
        "source_type": "clean_text",
        "category": "utilities_news",
        "notes": "Place name (Bengaluru) and negation (ಇಲ್ಲ - no)"
    },
    {
        "sample_id": "trans_kn_02",
        "source_text": "ಕರ್ನಾಟಕ ಸರ್ಕಾರವು 10000 ಯುವಕರಿಗೆ ಉಚಿತ ಕೌಶಲ್ಯ ತರಬೇತಿ ನೀಡಲು ನಿರ್ಧರಿಸಿದೆ.",
        "source_language": "kn",
        "source_language_name": "Kannada",
        "reference_english": "The Karnataka government decided to provide free skill training to 10000 youth.",
        "category": "employment_news",
        "source_type": "clean_text",
        "notes": "Proper noun (Karnataka) and number (10000)"
    },
    {
        "sample_id": "trans_kn_03",
        "source_text": "ಕರಾವಳಿ ಜಿಲ್ಲೆಗಳಲ್ಲಿ ಮುಂದಿನ 24 ಗಂಟೆಗಳಲ್ಲಿ ಭಾರೀ ಮಳೆಯಾಗುವ ಸಾಧ್ಯತೆಯಿದೆ ಎಂದು ಹವಾಮಾನ ಇಲಾಖೆ ತಿಳಿಸಿದೆ.",
        "source_language": "kn",
        "source_language_name": "Kannada",
        "reference_english": "The Meteorological Department stated that heavy rain is likely in coastal districts during the next 24 hours.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Duration (24 hours) and weather forecast"
    },
    {
        "sample_id": "trans_kn_04",
        "source_text": "ಕಾವೇರಿ ಜಲಾಶಯದಿಂದ ತಮಿಳುನಾಡಿಗೆ 15000 ಕ್ಯೂಸೆಕ್ ನೀರು ಬಿಡಲಾಗಿದೆ.",
        "source_language": "kn",
        "source_language_name": "Kannada",
        "reference_english": "15000 cusecs of water was released from the Cauvery reservoir to Tamil Nadu.",
        "category": "water_news",
        "source_type": "clean_text",
        "notes": "Entities (Cauvery, Tamil Nadu) and number (15000)"
    },
    {
        "sample_id": "trans_kn_05",
        "source_text": "ರಾಜ್ಯದ 150 ತಾಲೂಕುಗಳನ್ನು ಬರಪೀಡಿತ ಪ್ರದೇಶಗಳೆಂದು ಸರ್ಕಾರ ಘೋಷಿಸಿದೆ.",
        "source_language": "kn",
        "source_language_name": "Kannada",
        "reference_english": "The government declared 150 taluks of the state as drought-affected areas.",
        "category": "agriculture_news",
        "source_type": "clean_text",
        "notes": "Quantity (150 taluks)"
    },
    {
        "sample_id": "trans_kn_06",
        "source_text": "ಶಾಲಾ ಮಕ್ಕಳಿಗೆ ಪೌಷ್ಟಿಕ ಆಹಾರ ವಿತರಣೆ ಯೋಜನೆಯನ್ನು ವಿಸ್ತರಿಸಲಾಗುವುದು ಎಂದು ಸಚಿವರು ಹೇಳಿದರು.",
        "source_language": "kn",
        "source_language_name": "Kannada",
        "reference_english": "The Minister said that the nutritious food distribution scheme for school children will be expanded.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "Government scheme policy"
    },

    # ── 7. BENGALI (6 samples) ──
    {
        "sample_id": "trans_bn_01",
        "source_text": "কলকাতা ও পার্শ্ববর্তী জেলাগুলিতে আগামী ২৪ ঘণ্টায় ভারী বৃষ্টির সতর্কতা জারি করা হয়েছে।",
        "source_language": "bn",
        "source_language_name": "Bengali",
        "reference_english": "A heavy rain warning was issued for Kolkata and neighboring districts for the next 24 hours.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Place name (Kolkata) and time (24 hours)"
    },
    {
        "sample_id": "trans_bn_02",
        "source_text": "মুখ্যমন্ত্রী ঘোষণা করেছেন যে কৃষকদের জন্য কোনো কর বৃদ্ধি করা হবে না।",
        "source_language": "bn",
        "source_language_name": "Bengali",
        "reference_english": "The Chief Minister announced that no tax increase will be made for farmers.",
        "category": "governance_news",
        "source_type": "clean_text",
        "notes": "Negation (কোনো কর বৃদ্ধি করা হবে না - no tax increase)"
    },
    {
        "sample_id": "trans_bn_03",
        "source_text": "স্বাস্থ্য দপ্তর জানিয়েছে রাজ্যে ডেঙ্গু সংক্রমণ সম্পূর্ণ নিয়ন্ত্রণে রয়েছে।",
        "source_language": "bn",
        "source_language_name": "Bengali",
        "reference_english": "The Health Department stated that dengue transmission is completely under control in the state.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "Health official statement"
    },
    {
        "sample_id": "trans_bn_04",
        "source_text": "পশ্চিমবঙ্গে নতুন শিল্প বিনিয়োগের জন্য ৫০০০ কোটি টাকার প্রকল্প অনুমোদিত হয়েছে।",
        "source_language": "bn",
        "source_language_name": "Bengali",
        "reference_english": "A project of 5000 crore rupees was approved for new industrial investment in West Bengal.",
        "category": "business_news",
        "source_type": "clean_text",
        "notes": "Proper noun (West Bengal) and quantity (5000 crore)"
    },
    {
        "sample_id": "trans_bn_05",
        "source_text": "কলকাতা মেট্রোর নতুন সম্প্রসারিত রুটে ট্রেন চলাচল আনুষ্ঠানিকভাবে শুরু হয়েছে।",
        "source_language": "bn",
        "source_language_name": "Bengali",
        "reference_english": "Train operations officially began on the newly extended route of Kolkata Metro.",
        "category": "infrastructure_news",
        "source_type": "clean_text",
        "notes": "Named entity (Kolkata Metro)"
    },
    {
        "sample_id": "trans_bn_06",
        "source_text": "মাধ্যমিক পরীক্ষার ফলাফল আগামী শুক্রবার সকাল ১০টায় প্রকাশিত হবে।",
        "source_language": "bn",
        "source_language_name": "Bengali",
        "reference_english": "Secondary examination results will be published next Friday at 10 AM.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "Time specification (next Friday at 10 AM)"
    },

    # ── 8. MARATHI (6 samples) ──
    {
        "sample_id": "trans_mr_01",
        "source_text": "मुंबई आणि उपनगरांमध्ये आज मुसळधार पावसाचा इशारा हवामान खात्याने दिला आहे.",
        "source_language": "mr",
        "source_language_name": "Marathi",
        "reference_english": "The Meteorological Department has issued a heavy rain warning for Mumbai and the suburbs today.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Place name (Mumbai) and weather bulletin"
    },
    {
        "sample_id": "trans_mr_02",
        "source_text": "शेतकऱ्यांच्या वीज बिलात कोणतीही वाढ केली जाणार नाही असे उपमुख्यमंत्र्यांनी सांगितले.",
        "source_language": "mr",
        "source_language_name": "Marathi",
        "reference_english": "The Deputy Chief Minister said that no increase will be made in farmers' electricity bills.",
        "category": "energy_news",
        "source_type": "clean_text",
        "notes": "Negation (कोणतीही वाढ केली जाणार नाही - no increase will be made)"
    },
    {
        "sample_id": "trans_mr_03",
        "source_text": "पुणे मेट्रोच्या नवीन टप्प्याचे उद्घाटन २५ ऑक्टोबर २०२६ रोजी होणार आहे.",
        "source_language": "mr",
        "source_language_name": "Marathi",
        "reference_english": "The inauguration of the new phase of Pune Metro will take place on 25 October 2026.",
        "category": "infrastructure_news",
        "source_type": "clean_text",
        "notes": "Date (25 October 2026) and place name (Pune Metro)"
    },
    {
        "sample_id": "trans_mr_04",
        "source_text": "महाराष्ट्र सरकारने आरोग्य क्षेत्रासाठी १०००० कोटी रुपयांची तरतूद केली आहे.",
        "source_language": "mr",
        "source_language_name": "Marathi",
        "reference_english": "The Maharashtra government has made a provision of 10000 crore rupees for the health sector.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "Proper noun (Maharashtra) and number (10000 crore)"
    },
    {
        "sample_id": "trans_mr_05",
        "source_text": "एसटी महामंडळाने सणासुदीच्या काळात ५०० अतिरिक्त बसेस सोडण्याचा निर्णय घेतला.",
        "source_language": "mr",
        "source_language_name": "Marathi",
        "reference_english": "The ST Corporation decided to run 500 extra buses during the festival season.",
        "category": "transport_news",
        "source_type": "clean_text",
        "notes": "Number (500) and transit entity"
    },
    {
        "sample_id": "trans_mr_06",
        "source_text": "विद्यापीठाने सर्व पदवी परीक्षांचे निकाल वेळेवर जाहीर करण्याचे आश्वासन दिले आहे.",
        "source_language": "mr",
        "source_language_name": "Marathi",
        "reference_english": "The University has promised to announce all degree examination results on time.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "Academic announcement"
    },

    # ── 9. GUJARATI (6 samples) ──
    {
        "sample_id": "trans_gu_01",
        "source_text": "અમદાવાદ અને સુરતમાં આગામી ત્રણ દિવસ ભારે વરસાદની હવામાન વિભાગે આગાહી કરી છે.",
        "source_language": "gu",
        "source_language_name": "Gujarati",
        "reference_english": "The Meteorological Department has forecast heavy rain in Ahmedabad and Surat for the next three days.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Place names (Ahmedabad, Surat) and duration (three days)"
    },
    {
        "sample_id": "trans_gu_02",
        "source_text": "સરકારે સ્પષ્ટ કર્યું કે દૂધના ભાવમાં કોઈ વધારો કરવામાં આવ્યો નથી.",
        "source_language": "gu",
        "source_language_name": "Gujarati",
        "reference_english": "The government clarified that no increase was made in milk prices.",
        "category": "governance_news",
        "source_type": "clean_text",
        "notes": "Negation (કોઈ વધારો કરવામાં આવ્યો નથી - no increase)"
    },
    {
        "sample_id": "trans_gu_03",
        "source_text": "ગુજરાત સરકારે નવા બંદરોના વિકાસ માટે ૪૦૦૦ કરોડ રૂપિયાની યોજના મંજૂર કરી.",
        "source_language": "gu",
        "source_language_name": "Gujarati",
        "reference_english": "The Gujarat government approved a scheme of 4000 crore rupees for the development of new ports.",
        "category": "infrastructure_news",
        "source_type": "clean_text",
        "notes": "Proper noun (Gujarat) and numeric measure (4000 crore)"
    },
    {
        "sample_id": "trans_gu_04",
        "source_text": "ગાંધીનગરમાં આયોજિત વાઇબ્રન્ટ ગુજરાત સમિટમાં ૧૦૦થી વધુ દેશો ભાગ લેશે.",
        "source_language": "gu",
        "source_language_name": "Gujarati",
        "reference_english": "More than 100 countries will participate in the Vibrant Gujarat Summit organized in Gandhinagar.",
        "category": "business_news",
        "source_type": "clean_text",
        "notes": "Named event (Vibrant Gujarat) and quantity (100)"
    },
    {
        "sample_id": "trans_gu_05",
        "source_text": "નર્મદા ડેમની જળસપાટી ૧૩૮ મીટર સુધી પહોંચી ગઈ છે.",
        "source_language": "gu",
        "source_language_name": "Gujarati",
        "reference_english": "The water level of Narmada Dam has reached up to 138 meters.",
        "category": "water_news",
        "source_type": "clean_text",
        "notes": "Proper noun (Narmada Dam) and number (138)"
    },
    {
        "sample_id": "trans_gu_06",
        "source_text": "વિદ્યાર્થીઓને ડિજિટલ શિક્ષણ માટે મફત ટેબ્લેટનું વિતરણ કરવામાં આવ્યું.",
        "source_language": "gu",
        "source_language_name": "Gujarati",
        "reference_english": "Free tablets were distributed to students for digital education.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "Student welfare policy"
    },

    # ── 10. PUNJABI (6 samples) ──
    {
        "sample_id": "trans_pa_01",
        "source_text": "ਮੌਸਮ ਵਿਭਾਗ ਨੇ ਪੰਜਾਬ ਦੇ ਕਈ ਜ਼ਿਲ੍ਹਿਆਂ ਵਿੱਚ ਭਾਰੀ ਮੀਂਹ ਦਾ ਅਲਰਟ ਜਾਰੀ ਕੀਤਾ ਹੈ।",
        "source_language": "pa",
        "source_language_name": "Punjabi",
        "reference_english": "The Meteorological Department has issued a heavy rain alert in several districts of Punjab.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Place name (Punjab) and weather bulletin"
    },
    {
        "sample_id": "trans_pa_02",
        "source_text": "ਮੁੱਖ ਮੰਤਰੀ ਨੇ ਕਿਹਾ ਕਿ ਕਿਸਾਨਾਂ ਨੂੰ ਬਿਜਲੀ ਸਬਸਿਡੀ ਵਿੱਚ ਕੋਈ ਕਟੌਤੀ ਨਹੀਂ ਕੀਤੀ ਜਾਵੇਗੀ।",
        "source_language": "pa",
        "source_language_name": "Punjabi",
        "reference_english": "The Chief Minister said that no reduction will be made in electricity subsidy for farmers.",
        "category": "agriculture_news",
        "source_type": "clean_text",
        "notes": "Negation (ਕੋਈ ਕਟੌਤੀ ਨਹੀਂ ਕੀਤੀ ਜਾਵੇਗੀ - no reduction will be made)"
    },
    {
        "sample_id": "trans_pa_03",
        "source_text": "ਪੰਜਾਬ ਸਰਕਾਰ ਨੇ ੫੦੦੦ ਨਵੀਆਂ ਪੁਲਿਸ ਭਰਤੀਆਂ ਦਾ ਐਲਾਨ ਕੀਤਾ ਹੈ।",
        "source_language": "pa",
        "source_language_name": "Punjabi",
        "reference_english": "The Punjab government has announced 5000 new police recruitments.",
        "category": "governance_news",
        "source_type": "clean_text",
        "notes": "Numeric quantity (5000)"
    },
    {
        "sample_id": "trans_pa_04",
        "source_text": "ਅੰਮ੍ਰਿਤਸਰ ਹਵਾਈ ਅੱਡੇ ਤੋਂ ਕੈਨੇਡਾ ਲਈ ਨਵੀਆਂ ਸਿੱਧੀਆਂ ਉਡਾਣਾਂ ਸ਼ੁਰੂ ਹੋਣਗੀਆਂ।",
        "source_language": "pa",
        "source_language_name": "Punjabi",
        "reference_english": "New direct flights to Canada will begin from Amritsar airport.",
        "category": "transport_news",
        "source_type": "clean_text",
        "notes": "Place names (Amritsar, Canada)"
    },
    {
        "sample_id": "trans_pa_05",
        "source_text": "ਸਿਹਤ ਵਿਭਾਗ ਨੇ ਪੇਂਡੂ ਖੇਤਰਾਂ ਵਿੱਚ ੧੦੦ ਨਵੇਂ ਆਮ ਆਦਮੀ ਕਲੀਨਿਕ ਖੋਲ੍ਹਣ ਦੀ ਤਿਆਰੀ ਕੀਤੀ ਹੈ।",
        "source_language": "pa",
        "source_language_name": "Punjabi",
        "reference_english": "The Health Department has prepared to open 100 new Aam Aadmi Clinics in rural areas.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "Quantity (100) and program name"
    },
    {
        "sample_id": "trans_pa_06",
        "source_text": "ਸਕੂਲ ਸਿੱਖਿਆ ਬੋਰਡ ਨੇ ਸਾਲਾਨਾ ਪ੍ਰੀਖਿਆਵਾਂ ਦੀਆਂ ਤਰੀਕਾਂ ਦਾ ਐਲਾਨ ਕਰ ਦਿੱਤਾ ਹੈ।",
        "source_language": "pa",
        "source_language_name": "Punjabi",
        "reference_english": "The School Education Board has announced the dates for annual examinations.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "Academic schedule announcement"
    },

    # ── 11. URDU (6 samples) ──
    {
        "sample_id": "trans_ur_01",
        "source_text": "محکمہ موسمیات نے شمالی اضلاع میں شدید بارش کی پیش گوئی کی ہے۔",
        "source_language": "ur",
        "source_language_name": "Urdu",
        "reference_english": "The Meteorological Department has predicted heavy rain in the northern districts.",
        "category": "weather_news",
        "source_type": "clean_text",
        "notes": "Weather forecast announcement"
    },
    {
        "sample_id": "trans_ur_02",
        "source_text": "وزیر صحت نے تصدیق کی کہ اسپتالوں میں ادویات کی کوئی کمی نہیں ہے۔",
        "source_language": "ur",
        "source_language_name": "Urdu",
        "reference_english": "The Health Minister confirmed that there is no shortage of medicines in hospitals.",
        "category": "health_news",
        "source_type": "clean_text",
        "notes": "Negation (کوئی کمی نہیں ہے - no shortage)"
    },
    {
        "sample_id": "trans_ur_03",
        "source_text": "حکومت نے تعلیمی وظائف کے لیے ۵۰ کروڑ روپے مختص کیے ہیں۔",
        "source_language": "ur",
        "source_language_name": "Urdu",
        "reference_english": "The government has allocated 50 crore rupees for educational scholarships.",
        "category": "education_news",
        "source_type": "clean_text",
        "notes": "Financial quantity (50 crore)"
    },
    {
        "sample_id": "trans_ur_04",
        "source_text": "دہلی اور لکھنؤ کے درمیان نئی تیز رفتار ریل سروس اگلے ماہ شروع ہوگی۔",
        "source_language": "ur",
        "source_language_name": "Urdu",
        "reference_english": "A new high-speed rail service between Delhi and Lucknow will begin next month.",
        "category": "transport_news",
        "source_type": "clean_text",
        "notes": "Place names (Delhi, Lucknow)"
    },
    {
        "sample_id": "trans_ur_05",
        "source_text": "شہری ترقی کے ادارے نے ۲۵ نئے پارکوں کی تعمیر کا منصوبہ منظور کیا۔",
        "source_language": "ur",
        "source_language_name": "Urdu",
        "reference_english": "The Urban Development Authority approved a project to construct 25 new parks.",
        "category": "infrastructure_news",
        "source_type": "clean_text",
        "notes": "Quantity (25)"
    },
    {
        "sample_id": "trans_ur_06",
        "source_text": "بینک نے قرض کی شرح سود میں کوئی اضافہ نہیں کیا ہے۔",
        "source_language": "ur",
        "source_language_name": "Urdu",
        "reference_english": "The bank has not made any increase in the loan interest rate.",
        "category": "financial_news",
        "source_type": "clean_text",
        "notes": "Negation (کوئی اضافہ نہیں کیا ہے - has not made any increase)"
    },

    # ── 12. OCR-DERIVED SAMPLES (from Phase 3/4) (3 samples) ──
    {
        "sample_id": "trans_ocr_ta_01",
        "source_text": "சென்னை வானிலை ஆய்வு மையம்‌ இன்று மழை எச்சரிக்கை விடுத்துள்ளது.",
        "source_language": "ta",
        "source_language_name": "Tamil",
        "reference_english": "The Chennai Meteorological Center has issued a rain warning today.",
        "category": "weather_news",
        "source_type": "ocr_derived",
        "notes": "Phase 3/4 OCR output from tamil_news_sample.png (contains U+200C)"
    },
    {
        "sample_id": "trans_ocr_en_01",
        "source_text": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "category": "ocr_sample",
        "source_type": "ocr_derived",
        "notes": "Phase 3/4 OCR output from english_clear.png"
    },
    {
        "sample_id": "trans_ocr_en_noisy",
        "source_text": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "source_language": "en",
        "source_language_name": "English",
        "reference_english": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.",
        "category": "ocr_sample",
        "source_type": "ocr_derived",
        "notes": "Phase 3/4 OCR output from english_noisy.png"
    },

    # ── 13. EDGE CASES (6 samples) ──
    {
        "sample_id": "trans_edge_empty",
        "source_text": "",
        "source_language": "none",
        "source_language_name": "None",
        "reference_english": "",
        "category": "edge_case",
        "source_type": "edge_case",
        "notes": "Empty input string"
    },
    {
        "sample_id": "trans_edge_whitespace",
        "source_text": "   \n\t   ",
        "source_language": "none",
        "source_language_name": "None",
        "reference_english": "",
        "category": "edge_case",
        "source_type": "edge_case",
        "notes": "Whitespace-only input"
    },
    {
        "sample_id": "trans_edge_numeric",
        "source_text": "1234567890 2026",
        "source_language": "none",
        "source_language_name": "None",
        "reference_english": "",
        "category": "edge_case",
        "source_type": "edge_case",
        "notes": "Numeric-only input"
    },
    {
        "sample_id": "trans_edge_punct",
        "source_text": "??? !!! --- ... @@@ ###",
        "source_language": "none",
        "source_language_name": "None",
        "reference_english": "",
        "category": "edge_case",
        "source_type": "edge_case",
        "notes": "Punctuation-only input"
    },
    {
        "sample_id": "trans_edge_unsupported",
        "source_text": "Bonjour tout le monde, comment allez-vous aujourd'hui?",
        "source_language": "fr",
        "source_language_name": "French",
        "reference_english": "Hello everyone, how are you today?",
        "category": "edge_case",
        "source_type": "edge_case",
        "notes": "Unsupported language (French) to test error handling"
    },
    {
        "sample_id": "trans_edge_short",
        "source_text": "ha",
        "source_language": "none",
        "source_language_name": "None",
        "reference_english": "",
        "category": "edge_case",
        "source_type": "edge_case",
        "notes": "Extreme short string under threshold"
    },
]


def generate():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "source_text", "source_language", "source_language_name",
        "reference_english", "category", "source_type", "notes"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(SAMPLES)
    print(f"Generated {len(SAMPLES)} translation evaluation samples in {OUTPUT_CSV}")


if __name__ == "__main__":
    generate()
