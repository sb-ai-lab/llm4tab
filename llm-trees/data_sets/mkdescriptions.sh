#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  bank_credit_scoring \
  callcenter \
  crimes_arrest \
  extrovert \
  machine \
  postpartum \
  reading \
  stars

cat > bank_credit_scoring/feature_description.txt <<'EOF'
Features:
Debt: client's current debt amount.
Overdue days: number of overdue days.
Initial limit: initial credit limit.
BIRTHDATE: client's date of birth represented as Unix timestamp after datetime-to-float conversion.
INCOME: client's income.
PDN: probability of default indicator.
SEX: client's sex (Male / Female).
EDU: client's education level (High school / University / College / Incomplete, University).
TERM: loan term (24 / 36 / 60).
Credit history rating: client's credit history rating (C3 / A1 / B3 / B1 / C1 / D3 / B2 / A3 / A2 / D2).
LV_AREA: client's region or area of residence (g. Minsk / MINSKAJa / BRESTSKAJa / Gomel'skaja oblast' / Brestskaja oblast' / GRODNENSKAJa / MINSKAJa OBLAST' / Minskaja oblast' / Vitebskaja oblast' / VITEBSKAJa / etc.).
LV_SETTLEMENTNAME: client's settlement or city of residence (MINSK / Minsk / PRUSY / PINSK / SVETLOGORSK / Oktjabr' / RAJTsA / MIHALOVO / SMOLJaNY / BARANOVIChI / etc.).
INDUSTRYNAME: client's employment industry (Production / Medicine and healthcare / Finance and Insurance / Construction / Agriculture / Retired / Transport/Logistic / Science and Education / Trade / Ministry of Internal Affairs/Ministry of Emergency Situations/Ministry of Defense / etc.).
UNDERAGECHILDRENCOUNT: number of underage children (0 / 1 / 2 / 3).
FAMILYSTATUS: family or marital status code (1 / 2).

Target variable:
target: credit scoring class (0 = low credit scoring / 1 = high credit scoring)
EOF

cat > callcenter/feature_description.txt <<'EOF'
Features:
Call_center_employee_communication_skills_evaluation: client's evaluation of call center employee communication skills.
Satisfaction_with_the_doctors_response: client's satisfaction with the doctor's response.
How_often_do_you_receive_call_center_news: frequency of receiving call center news.
Gender: client's gender (Female / Male).
Age: client's age group (40-49 / 30-39 / 18-24 / 25-29 / 60 and above / Under 18 / 50-59).
Education: client's education level (Bachelor / Diploma / High school / Less than high school / Master / Doctor / Other).
Nationality: client's nationality group (Kuwaiti / Other nationality / Unspecified).
Marital_Status: client's marital status (Married / Single / Widow / Widower).
Governorate: client's governorate or region (Farwaniya / Capital / Around / AlAhmadi / Jahra / Mubarak AlKabir).
Aware_151: whether the client is aware of service 151 (Yes / No).
Call_151: whether the client has called service 151 (Yes / No).
The_reason_of_the_call: main reason for the call (Vaccination / Inquiries / Inquiries_COVID19 / Medical_Consultation / Complaints).

Target variable:
target: satisfaction class (0 = dissatisfied / 1 = satisfied)
EOF

cat > crimes_arrest/feature_description.txt <<'EOF'
Features:
Date: date and time of the crime event represented as Unix timestamp after datetime-to-float conversion.
Beat: police beat code.
District: police district code.
Ward: ward code.
Community Area: Chicago community area code.
X Coordinate: projected x coordinate of the event location.
Y Coordinate: projected y coordinate of the event location.
Latitude: geographic latitude of the event location.
Longitude: geographic longitude of the event location.
Primary Type: primary crime type (BATTERY / THEFT / OTHER OFFENSE / CRIMINAL TRESPASS / ASSAULT / PUBLIC PEACE VIOLATION / WEAPONS VIOLATION / CRIMINAL SEXUAL ASSAULT / CRIMINAL DAMAGE / DECEPTIVE PRACTICE / etc.).
Description: detailed crime description (SIMPLE / OVER $500 / DOMESTIC BATTERY SIMPLE / TELEPHONE THREAT / RETAIL THEFT / $500 AND UNDER / TO LAND / VEHICLE TITLE / REGISTRATION OFFENSE / RECKLESS CONDUCT / TO STATE SUP LAND / etc.).
Location Description: location type where the crime occurred (NURSING / RETIREMENT HOME / STREET / GAS STATION / APARTMENT / CTA BUS / RESIDENCE - PORCH / HALLWAY / DEPARTMENT STORE / BAR OR TAVERN / SMALL RETAIL STORE / RESTAURANT / etc.).
Domestic: whether the crime was domestic (0 / 1).
FBI Code: FBI crime classification code (08B / 06 / 08A / 26 / 24 / 15 / 02 / 14 / 04B / 11 / etc.).
Year: year of the event.

Target variable:
target: arrest outcome (0 = no arrest / 1 = arrest made)
EOF

cat > extrovert/feature_description.txt <<'EOF'
Features:
social_connection_frequency: frequency of social interaction (Daily / Weekly / Never / Monthly / weekly / etc.).
gathering_size_preference: preferred size of social gatherings (Small gatherings / Both / Large events / Never).
valued_friendship_traits: traits most valued in friendship (Support / Loyalty / Honesty / All / All the three and trust / Understanding person / etc.).
conflict_resolution_style: preferred conflict resolution style (Talk it out / Apologize first / Avoid conflict / Let time fix it).
family_support_level: level of family support (Yes / Sometimes / No / Never / yes / etc.).
social_battery_recharge: preferred way to recharge socially (Talking to someone / Alone time / None of the above / Social media).
social_orientation_type: self-identified social orientation (Introvert / Ambivert / Extrovert / Not sure).
social_anxiety_level: level of social anxiety (Avoid it / Neutral / Excited / Nervous).
emotional_expression_method: way of expressing emotions (Stay silent / Talk openly / Talk openly, Express emotion strongly / Express emotion strongly / Other / etc.).
primary_anger_sources: main anger triggers (Injustice / Being interrupted / Selfishness / Unnecessary criticism).

Target variable:
target: friendship-maintenance class (0 = uses texting/social media or rarely keeps in touch / 1 = maintains long-distance friendships through calls)
EOF

cat > machine/feature_description.txt <<'EOF'
Features:
Temperature: machine temperature during operation.
Vibration: machine vibration level during operation.
Power_Usage: machine power consumption.
Humidity: humidity in the operating environment.
Machine_Type: machine type (Mill / Lathe / Drill).

Target variable:
target: machine status (0 = normal operation / 1 = at risk of failure)
EOF

cat > postpartum/feature_description.txt <<'EOF'
Features:
Age: patient's age.
Residence: place of residence (City / Village).
Education Level: patient's education level (University / College / Primary School / High School / No / High school / Primary school / etc.).
Marital status: patient's marital status (Married / Divorced).
Occupation before latest pregnancy: patient's occupation before latest pregnancy (Student / Doctor / Service / Housewife / Teacher / Other / Business / House wife / etc.).
Monthly income before latest pregnancy: patient's monthly income before latest pregnancy (No / 10000 to 20000 / More than 30000 / Less than 5000 / 5000 to 10000 / 20000 to 30000).
Occupation After Your Latest Childbirth: patient's occupation after latest childbirth (Student / Doctor / Service / Housewife / Teacher / Business / Other / House wife / etc.).
Current monthly income: patient's current monthly income (No / 10000 to 20000 / More than 30000 / Less than 5000 / 5000 to 10000 / 20000 to 30000).
Husband's education level: husband's education level (University / No / College / Primary School / High School / High school / Primary school / etc.).
Husband’s monthly income: husband's monthly income (More than 30000 / No / 10000 to 20000 / 5000 to 10000 / 20000 to 30000 / Less than 5000).
Addiction: addiction status (No / Smoking / Drinking / Drugs).
Total children: total number of children (One / Two / More than two / More than Two / etc.).
Disease before pregnancy: disease status before pregnancy (No / Non-Chronic Disease / Chronic Disease / 1.2 / etc.).
History of pregnancy loss: pregnancy loss history (No / Miscarriage / Still-born Delivery / Still-born delivery / etc.).
Family type: family structure (Nuclear / Joint).
Number of household members: number of household members (6 to 8 / 2 to 5 / 9 or more).
Relationship with the in-laws: relationship with the in-laws (Neutral / Good / Bad / Friendly / Poor).
Relationship with husband: relationship with husband (Good / Neutral / Bad / Friendly / Poor).
Relationship with the newborn: relationship with the newborn (Good / Neutral / Bad / Very good).
Relationship between father and newborn: relationship between father and newborn (Good / Neutral / Bad / Very good).
Feeling about motherhood: feeling about motherhood (Neutral / Happy / Sad).
Recieved Support: level of received support (High / Medium / Low).
Need for Support: level of needed support (Medium / Low / No / High).
Major changes or losses during pregnancy: whether major changes or losses happened during pregnancy (Yes / No).
Abuse: whether abuse was experienced (Yes / No).
Trust and share feelings: whether the patient trusts others and shares feelings (Yes / No).
Number of the latest pregnancy: pregnancy number (1 / 2 / 3 / 4 / 5 / 7 / 6).
Pregnancy length: pregnancy duration group (10 months / 9 months / 8 months / Less than 5 months / 7 months / 6 months).
Pregnancy plan: whether the pregnancy was planned (No / Yes).
Regular checkups: whether regular checkups were done (Yes / No).
Fear of pregnancy: whether fear of pregnancy was present (Yes / No).
Diseases during pregnancy: disease status during pregnancy (No / Chronic Disease / Non-Chronic Disease / Non-chronic disease / Chronic disease / etc.).
Age of newborn: age group of the newborn (6 months to 1 year / Older than 1.5 year / 0 to 6 months / 1 year to 1.5 year).
Age of immediate older children: age group of immediate older children (No / 13yr or more / 1yr to 3yr / 7yr to 12yr / 4yr to 6yr).
Mode of delivery: delivery type (Normal Delivery / Caesarean Section).
Gender of newborn: newborn's gender (Boy / Girl).
Birth compliancy: whether there were birth complications (No / Yes).
Breastfeed: whether the newborn is breastfed (Yes / No).
Newborn illness: whether the newborn has illness (No / Yes).
Worry about newborn: whether the patient worries about the newborn (Yes / No).
Relax/sleep when newborn is tended: whether the patient can relax or sleep when the newborn is tended by others (Yes / No).
Relax/sleep when the newborn is asleep: whether the patient can relax or sleep when the newborn is asleep (Yes / No).
Angry after latest child birth: whether the patient feels angry after latest childbirth (No / Yes).
Feeling for regular activities: feeling about regular activities (Worried / Tired / No / Afraid).

Target variable:
target: postpartum depression status (0 = does not have postpartum depression / 1 = has postpartum depression)
EOF

cat > reading/feature_description.txt <<'EOF'
Features:
Gender: student's gender (Male / Female).
Age: student's age (22 / 23 / 21 / 20 / 19 / 18).
Department: student's academic department (CSE / SWE / English / Pharmacy / EEE / BBA / NFE / Others / MCT / Textile / etc.).
Amount of time spent reading books per day: amount of time spent reading books per day.
Reason for reading books: main reason for reading books (Academic Purpose / Keep me informative / Entertainment Purpose / To increase my English vocabulary knowledge / etc.).
How to manage books for reading: how books are obtained or managed (E-books / Borrow / Others / Buy / Used-all).
Number of e-books you read last two years: number of e-books read in the last two years (11-15 / More than 20 / 1-5 / 6-10 / 16-20).
Number of printed books you read last two years: number of printed books read in the last two years (11-15 / 1-5 / 6-10 / More than 20 / 16-20).
Favorite books for reading to spend leisure time: preferred books for leisure reading (Biography / Non-Fiction / Others / Fiction / Politics).
What is the change that you fell after reading books: perceived effect after reading books (Increase Academic Performance / Increase Creative Thinking / Feel Relax / Increase Problem Solving Skill).
Who influenced you to read books: who influenced the student to read books (Teachers / Self-Decision / Friends / Parents).
Which language books do you like to read: preferred language for books (English / Bangla / Others).
Where do you read printed books: place where printed books are read (Home / Library / During Travel).
Do you read Newspaper: whether the student reads newspapers (No / Yes).
The format that you used for reading books: preferred reading format (Both / E-books / Printed).
Frequency of reading books: reading frequency (On everyday basis / A few times a week / A few times a month / Less than once a year / Once or twice a year).

Target variable:
target: reading preference class (0 = reads non-academic books / 1 = reads academic books most of the time)
EOF

cat > stars/feature_description.txt <<'EOF'
Features:
Vmag: apparent visual magnitude of the star.
Plx: parallax of the star.
e_Plx: parallax measurement error.
B-V: color index of the star.
Amag: absolute magnitude of the star.
SpType: spectral type of the star (G1V / F3V / F7IV / B8/B9V / K5V / F5V / B3III/IV / A4V+... / K3III / B9V / etc.).

Target variable:
target: star class (0 = dwarf star / 1 = giant star)
EOF
