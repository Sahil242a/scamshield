"""Coverage expansion: message TYPES the first versions never saw.

Both user-reported false positives (an itemised Airtel bill, a Hindi Jio promo) were message types absent
from training, so the model judged them by surface similarity to scams. The cheapest, most effective fix
for "unknown messages" is coverage: many kinds of real messages, each with several phrasings, in all three
languages. Written independently of data/generalization_set.csv (which measures exactly this).
"""

EXTRA_VALUES = {
    "veh": ["RJ14CB1234", "UP27AK4421", "DL8CAF5521", "MH12QW9087", "KA05MN3344", "HR26DK7788"],
    "pnr": ["4123567890", "2659081234", "8812340056", "6677120934"],
    "train": ["12015 Shatabdi", "12951 Rajdhani", "19665 Udaipur Exp", "12460 Intercity", "22691 Rajdhani"],
    "flight": ["6E 512", "AI 471", "QP 1342", "UK 818", "6E 2087"],
    "route": ["DEL-BOM", "JAI-BLR", "BOM-GOI", "HYD-DEL", "CCU-DEL"],
    "fund": ["HDFC Index Fund", "SBI Small Cap Fund", "Axis Bluechip Fund", "Nippon Large Cap Fund", "Mirae Asset ELSS"],
    "city": ["Jaipur", "Lucknow", "Pune", "Indore", "Kanpur", "Bhopal", "Delhi", "Patna"],
    "course": ["DBMS", "Operating Systems", "Machine Learning", "Physics", "Maths-II"],
    "insurer": ["LIC", "HDFC Life", "Star Health", "ICICI Lombard", "Tata AIA"],
    "ott": ["Hotstar", "SonyLIV", "Prime Video", "Netflix", "Zee5", "JioCinema"],
    "school": ["DPS", "Kendriya Vidyalaya", "St. Mary's School", "Ryan International"],
    "lab": ["Dr Lal PathLabs", "Thyrocare", "SRL Diagnostics", "Metropolis"],
    "promo_code": ["SAVE50", "FIRST100", "FEST20", "NEWUSER", "TRYNOW"],
    "broker": ["Zerodha", "Groww", "Upstox", "Angel One"],
}

EXTRA_LEGIT = [
    # banking: mandates, statements, cheques, cards, security
    ("bank2", "en", "Dear Customer, e-mandate for Rs {small} towards {ott} has been registered on your {bank_short} card XX{last4}. To cancel, visit the bank app."),
    ("bank2", "en", "Your {bank} Credit Card statement is generated. Total amount due Rs {amt}, min due Rs {fee}. Pay by {date} via the {bank_short} app."),
    ("bank2", "en", "Cheque {last4} for Rs {amt} presented in your A/c XX{last4} has been returned unpaid. Contact your branch for details. -{bank_short}"),
    ("bank2", "en", "Your {bank_short} Debit Card XX{last4} has been dispatched via India Post. Activate it from the mobile banking app on receipt."),
    ("bank2", "en", "You have successfully changed your {bank_short} NetBanking password on {date}. If you did not do this, call 1800 425 3800."),
    ("bank2", "en", "Fixed Deposit of Rs {amt} booked in your {bank} account. Maturity date {date}. -{bank_short}"),
    ("bank2", "en", "Money received! Rs {amt} from {name} credited to your account via UPI. Ref {ref}. -{app}"),
    ("bank2", "hinglish", "{name} ne aapko Rs {amt} bheje hai. Paise aapke bank account me jama ho gaye hai. -{app}"),
    ("bank2", "hi", "आपके {bank_short} क्रेडिट कार्ड का बिल {amt} रुपये है। अंतिम तिथि {date}। बैंक ऐप से भुगतान करें।"),
    ("bank2", "hi", "{name} ने आपको {amt} रुपये भेजे हैं। राशि आपके खाते में जमा हो गई है।"),
    # insurance, EPF, investments
    ("finance", "en", "{insurer}: Renewal premium of Rs {amt} for your policy is due on {date}. Pay on our official website or app to keep cover active."),
    ("finance", "en", "{insurer}: Your policy document has been emailed to you. Thank you for choosing us."),
    ("finance", "en", "Your EPF account has been credited with Rs {amt} as contribution for {month}. Check balance on the UMANG app."),
    ("finance", "en", "SIP instalment of Rs {mid} for {fund} processed successfully on {date}. -{broker}"),
    ("finance", "en", "Contract note for your trades on {date} has been sent to your registered email. -{broker}"),
    ("finance", "en", "Your redemption of Rs {amt} from {fund} will be credited to your bank within 3 working days."),
    ("finance", "hinglish", "Aapki {insurer} policy ka premium Rs {amt} {date} tak bharna hai. Official app ya website se bhugtan karein."),
    ("finance", "hi", "आपके पीएफ खाते में {month} माह का अंशदान {amt} रुपये जमा हुआ है।"),
    # travel
    ("travel", "en", "PNR {pnr}: {train} is running late by {mins} minutes. Please plan accordingly. -IRCTC"),
    ("travel", "en", "PNR {pnr} CNF. {train} on {date}, coach B2, berth 34. Happy journey!"),
    ("travel", "en", "Your flight {flight} {route} on {date} is now boarding from Gate 12. -IndiGo"),
    ("travel", "en", "Web check-in for flight {flight} ({route}) is open. Check in on the airline app and get your boarding pass."),
    ("travel", "en", "Rs {small} has been deducted from your FASTag for vehicle {veh} at a toll plaza. Avl bal Rs {bal}."),
    ("travel", "en", "Your FASTag balance is low (Rs {small}). Recharge through your bank app or any UPI app to avoid double toll."),
    ("travel", "hinglish", "Aapki train {train} {mins} minute late chal rahi hai. PNR {pnr}."),
    ("travel", "hinglish", "FASTag se Rs {small} kata, gaadi {veh}. Bacha balance Rs {bal}."),
    ("travel", "hi", "पीएनआर {pnr}: आपकी ट्रेन {train} {mins} मिनट देरी से चल रही है।"),
    ("travel", "hi", "आपके फास्टैग से वाहन {veh} के लिए {small} रुपये कटे। शेष राशि {bal} रुपये।"),
    # OTP handover / app OTPs
    ("handover", "en", "Your delivery OTP is {last4}. Share it with the delivery partner only at the time of delivery. -Flipkart"),
    ("handover", "en", "Ride OTP {last4}. Give this to your captain to start the ride. -Rapido"),
    ("handover", "en", "Technician {name} will visit today between 2-4 PM. Share service OTP {last4} after the job is done. -Urban Company"),
    ("handover", "hinglish", "Aapka delivery OTP {last4} hai. Order milne par hi delivery boy ko batayein."),
    ("handover", "hi", "आपका डिलीवरी ओटीपी {last4} है। ऑर्डर मिलने पर ही डिलीवरी एजेंट को बताएं।"),
    # security alerts from apps (legit "if not you" with no phone number)
    ("security", "en", "New sign-in to your Google Account on a Windows device. If this was you, no action needed. Otherwise, review activity in your account settings."),
    ("security", "en", "Your {app} password was changed. If you didn't do this, reset it from the app's login screen."),
    ("security", "en", "Someone tried to log in to your Instagram from {city}. We blocked it. Review login activity in Settings."),
    ("security", "hinglish", "Aapke {app} account me naye device se login hua hai. Agar yeh aap nahi the to app me jaakar password badlein."),
    # government / civic
    ("govt_legit", "en", "Your Aadhaar update request has been processed. Download updated e-Aadhaar from the official myAadhaar portal."),
    ("govt_legit", "en", "Reminder from the Income Tax Department: file your return before the due date at incometax.gov.in. Ignore if already filed."),
    ("govt_legit", "en", "Your PAN card application has been dispatched. Track status on the official NSDL/UTI website."),
    ("govt_legit", "en", "Property tax for FY 2026-27 is due. Pay online on the municipal corporation portal before {date} to avoid penalty."),
    ("govt_legit", "en", "Water supply will be interrupted in your area on {date} from 10 AM to 4 PM due to pipeline work. -Jal Board"),
    ("govt_legit", "hinglish", "Kal {city} ke kuch ilakon me subah 9 se 1 baje tak bijli nahi rahegi, maintenance kaam ke karan. -{board}"),
    ("govt_legit", "hi", "जल विभाग: कल सुबह 10 से शाम 4 बजे तक आपके क्षेत्र में पानी की आपूर्ति बंद रहेगी।"),
    ("govt_legit", "hi", "नगर निगम: संपत्ति कर {date} तक जमा करें। ऑनलाइन भुगतान की सुविधा उपलब्ध है।"),
    ("govt_legit", "hi", "आपका आधार अपडेट अनुरोध सफलतापूर्वक पूरा हो गया है।"),
    # health, education, jobs
    ("health_edu", "en", "Your {lab} test report is ready. View it in the app using your registered mobile number."),
    ("health_edu", "en", "Appointment confirmed with Dr. {name} on {date} at {time} AM. Please arrive 15 minutes early."),
    ("health_edu", "en", "Results for {course} mid-term are published on the student portal. -Exam Section"),
    ("health_edu", "en", "{school}: School will remain closed on {date} on account of the festival."),
    ("health_edu", "en", "Dear Parent, fee for the next quarter is due by {date}. Pay at the school office or the school app."),
    ("health_edu", "en", "{company} has viewed your profile. 5 new jobs match your preferences. Open the app to apply."),
    ("health_edu", "en", "Your interview with {company} is confirmed for {date} at {time}. Join using the link in your email."),
    ("health_edu", "hinglish", "Kal {course} ka lab exam hai, sab apni file aur ID card le aana."),
    ("health_edu", "hi", "{school}: {date} को विद्यालय में अभिभावक-शिक्षक बैठक है। कृपया उपस्थित रहें।"),
    ("health_edu", "hi", "आपकी जांच रिपोर्ट तैयार है। लैब ऐप में देखें।"),
    # subscriptions, apps, promos with codes
    ("apps", "en", "Your {ott} subscription has been renewed for Rs {small}. Enjoy watching!"),
    ("apps", "en", "Your {ott} plan expires on {date}. Renew in the app to keep watching."),
    ("apps", "en", "Use code {promo_code} to get {pct}% off on your next order. Valid till {date}. -{app}"),
    ("apps", "en", "Cashback of Rs {small} credited to your {app} wallet for your last order. Use it on your next purchase."),
    ("apps", "en", "Your {app} order #{order} has been cancelled and the refund of Rs {amt} will be processed in 5-7 days."),
    ("apps", "hinglish", "{app} pe aaj {pct}% off! Code {promo_code} lagao aur order karo."),
    ("apps", "hi", "आपकी {ott} सदस्यता {date} को समाप्त हो रही है। ऐप में नवीनीकरण करें।"),
    # more personal messages (EN / Hinglish / Hindi)
    ("chat2", "en", "Hey {name}, can you pick up the cake on your way? I'll pay you back."),
    ("chat2", "en", "Landed in {city}. Will call once I reach the hotel."),
    ("chat2", "en", "Sending you Rs {small} for the movie tickets, check GPay."),
    ("chat2", "en", "The electrician will come tomorrow at {time}, please keep the door open."),
    ("chat2", "hinglish", "{name} bhai, kal {city} jaa raha hu, kuch mangwana hai to bata"),
    ("chat2", "hinglish", "Maine {small} bhej diye hai, check kar lena aur bata dena"),
    ("chat2", "hinglish", "Aaj class me {course} ka quiz tha, tu kyu nahi aaya?"),
    ("chat2", "hinglish", "Didi, mummy bol rahi hai shaam ko jaldi aa jana, mehmaan aa rahe hai"),
    ("chat2", "hinglish", "Bhaiya ka phone band aa raha hai, unko bolna mujhe call karein"),
    ("chat2", "hi", "{name}, कल शाम को {time} बजे मिलते हैं।"),
    ("chat2", "hi", "मैं {city} पहुंच गया हूँ, चिंता मत करना।"),
    ("chat2", "hi", "बेटा, पापा ने {small} रुपये भेजे हैं, देख लेना।"),
]

EXTRA_SCAM = [
    # fake alerts with mobile callback numbers
    ("callback", "en", "ALERT: Rs {amt} has been withdrawn from your {bank} account. If this was not you, call our security desk at {phone} immediately."),
    ("callback", "en", "Your {app} order for {item} worth Rs {amt} is placed. Didn't order? Call customer support {phone} to cancel and get refund."),
    ("callback", "en", "A new payee was added to your {bank_short} account. If you did not do this, call {phone} now to block it."),
    ("callback", "hinglish", "Aapke {bank_short} account se Rs {amt} debit hue hai. Agar aapne nahi kiya to turant {phone} par call karein."),
    ("callback", "hi", "आपके {bank_short} खाते से {amt} रुपये डेबिट हुए हैं। अगर यह आपने नहीं किया तो {phone} पर कॉल करें।"),
    # account takeover: send the code back
    ("takeover", "en", "Hi, sorry, my WhatsApp code was sent to your number by mistake. Please forward me the 6 digit code you just got."),
    ("takeover", "en", "This is {name} from your college group, I'm locked out. A code came to your phone, can you send it to me?"),
    ("takeover", "hinglish", "Yaar mera code galti se tere number pe aa gaya, jo 6 digit aaya hai wo bhej de please"),
    ("takeover", "hi", "गलती से मेरा कोड आपके नंबर पर आ गया है, कृपया वह 6 अंकों का कोड मुझे भेज दें।"),
    # e-challan, govt schemes, subsidies, EPF
    ("govt_scam", "en", "Traffic e-challan of Rs {fee} issued for vehicle {veh}. Pay today to avoid licence suspension: {url}"),
    ("govt_scam", "en", "You are eligible for Rs {mid} under a government scheme. Complete registration with a fee of Rs {fee}: {url}"),
    ("govt_scam", "en", "Your PF withdrawal of Rs {amt} is stuck. Update KYC and bank details now: {url}"),
    ("govt_scam", "en", "Gas subsidy of Rs {small} not credited due to Aadhaar mismatch. Update here to receive it: {url}"),
    ("govt_scam", "hinglish", "Aapki gaadi {veh} ka challan Rs {fee} pending hai, aaj bharein warna licence radd: {url}"),
    ("govt_scam", "hinglish", "Sarkari yojana me aapka naam aaya hai, Rs {mid} paane ke liye registration fees Rs {fee} bhejein {upi}"),
    ("govt_scam", "hi", "आपके वाहन {veh} पर {fee} रुपये का चालान बकाया है। लाइसेंस निलंबन से बचने के लिए भुगतान करें {url}"),
    ("govt_scam", "hi", "सरकारी योजना में आपका नाम आया है। {mid} रुपये पाने के लिए पंजीकरण शुल्क {fee} रुपये जमा करें।"),
    # OLX / marketplace / army buyer
    ("market", "en", "Hello, I am from the Army, I want to buy your {item}. I will send money by QR code, please scan it and enter your PIN to get the payment."),
    ("market", "en", "I liked your bike ad. I'm in the CRPF and posted outside, I'll pay advance through Google Pay request, just approve it to receive."),
    ("market", "hinglish", "Sir main army me hu, aapka {item} lena hai. QR bhej raha hu, scan karke PIN daalo paise aa jayenge"),
    # data entry / review / crypto / trading
    ("task2", "en", "Online job: data entry from home, earn Rs {mid} weekly. Registration charge Rs {fee} only. Reply YES."),
    ("task2", "en", "Get paid Rs {small} per Google review. Daily earning up to Rs {mid}. Contact our HR on Telegram."),
    ("task2", "en", "Our AI trading bot makes {pct}% profit every day. Deposit Rs {mid} and withdraw anytime. Join the group: {url}"),
    ("task2", "hinglish", "Roz {mid} kamao, bas hotels ko rating dena hai. Pehle task free, phir thoda invest karna hoga. Telegram pe aao"),
    ("task2", "hi", "क्रिप्टो में निवेश करें और रोज {pct}% मुनाफा पाएं। अभी ग्रुप जॉइन करें {url}"),
    # extortion / fake relative arrested / fake officer
    ("extortion", "en", "Your private video will be shared with your family unless you transfer Rs {mid} within 2 hours. Contact {phone}."),
    ("extortion", "en", "Your son is in police custody after an accident. Send Rs {mid} for bail now and do not tell anyone, it will make it worse."),
    ("extortion", "hinglish", "Aapke bete ko police ne pakad liya hai, chhudane ke liye abhi {mid} bhejo, kisi ko mat batana"),
    ("extortion", "hi", "आपके बेटे को पुलिस ने पकड़ा है। अभी {mid} रुपये भेजें, किसी को मत बताना।"),
    # fake wallet / payment app restrictions
    ("wallet", "en", "Your {app} account is restricted due to unusual activity. Verify with your debit card details here: {url}"),
    ("wallet", "en", "Spin & Win: you won {gift}! Pay delivery charges of Rs {fee} to get it shipped. {url}"),
    ("wallet", "hinglish", "Aapka {app} account limit ho gaya hai, card details se verify karein: {url}"),
    # courier OTP ask
    ("courier2", "en", "{app} delivery: your parcel is waiting. Reply with the OTP sent to your phone so the agent can confirm the address."),
    ("courier2", "hinglish", "Courier boy bol raha hu, aapke phone pe OTP aaya hoga, address confirm karne ke liye bata dijiye"),
]
