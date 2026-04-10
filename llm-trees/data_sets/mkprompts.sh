#!/usr/bin/env bash
set -euo pipefail

create_prompt() {
  local dir="$1"
  local task="$2"
  local features="$3"

  mkdir -p "$dir"

  cat > "$dir/prompt.txt" <<EOF
I want you to induce a decision tree classifier based on features. I first give an example below.
Then, I provide you with Features and want you to build a decision tree with a maximum depth of 2 using the most important features.
The tree should classify $task.

Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)

Decision tree:
|--- petal width (cm) <= 0.80
||--- class: setosa
|--- petal width (cm) > 0.80
||--- petal width (cm) <= 1.75
|||--- class: versicolor
||--- petal width (cm) > 1.75
|||--- class: virginica

Features: $features

Decision Tree:
EOF
}

create_prompt \
  "bank_credit_scoring" \
  "credit scoring class (low credit scoring / high credit scoring)" \
  "Debt, Overdue days, Initial limit, BIRTHDATE, INCOME, PDN, SEX (Male / Female), EDU (High school / University / College / Incomplete, University), TERM (24 / 36 / 60), Credit history rating (C3 / A1 / B3 / B1 / C1 / D3 / B2 / A3 / A2 / D2), LV_AREA (g. Minsk / MINSKAJa / BRESTSKAJa / Gomel'skaja oblast' / Brestskaja oblast' / GRODNENSKAJa / MINSKAJa OBLAST' / Minskaja oblast' / Vitebskaja oblast' / VITEBSKAJa / etc.), LV_SETTLEMENTNAME (MINSK / Minsk / PRUSY / PINSK / SVETLOGORSK / Oktjabr' / RAJTsA / MIHALOVO / SMOLJaNY / BARANOVIChI / etc.), INDUSTRYNAME (Production / Medicine and healthcare / Finance and Insurance / Construction / Agriculture / Retired / Transport/Logistic / Science and Education / Trade / Ministry of Internal Affairs/Ministry of Emergency Situations/Ministry of Defense / etc.), UNDERAGECHILDRENCOUNT (0 / 1 / 2 / 3), FAMILYSTATUS (1 / 2)" \

create_prompt \
  "callcenter" \
  "satisfaction class (dissatisfied / satisfied)" \
  "Call_center_employee_communication_skills_evaluation, Satisfaction_with_the_doctors_response, How_often_do_you_receive_call_center_news, Gender (Female / Male), Age (18-24 / 25-29 / 30-39 / 40-49 / 50-59 / 60 and above / Under 18), Education (Bachelor / Diploma / High school / Less than high school / Master / Doctor / Other), Nationality (Kuwaiti / Other nationality / Unspecified), Marital_Status (Married / Single / Widow / Widower), Governorate (Farwaniya / Capital / Around / AlAhmadi / Jahra / Mubarak AlKabir), Aware_151 (Yes / No), Call_151 (Yes / No), The_reason_of_the_call (Vaccination / Inquiries / Inquiries_COVID19 / Medical_Consultation / Complaints)" \

create_prompt \
  "crimes_arrest" \
  "arrest outcome (no arrest / arrest made)" \
  "Date, Beat, District, Ward, Community Area, X Coordinate, Y Coordinate, Latitude, Longitude, Primary Type (BATTERY / THEFT / OTHER OFFENSE / CRIMINAL TRESPASS / ASSAULT / PUBLIC PEACE VIOLATION / WEAPONS VIOLATION / CRIMINAL SEXUAL ASSAULT / CRIMINAL DAMAGE / DECEPTIVE PRACTICE / etc.), Description (SIMPLE / OVER \$500 / DOMESTIC BATTERY SIMPLE / TELEPHONE THREAT / RETAIL THEFT / \$500 AND UNDER / TO LAND / VEHICLE TITLE / REGISTRATION OFFENSE / RECKLESS CONDUCT / TO STATE SUP LAND / etc.), Location Description (NURSING / RETIREMENT HOME / STREET / GAS STATION / APARTMENT / CTA BUS / RESIDENCE - PORCH / HALLWAY / DEPARTMENT STORE / BAR OR TAVERN / SMALL RETAIL STORE / RESTAURANT / etc.), Domestic (0 / 1), FBI Code (08B / 06 / 08A / 26 / 24 / 15 / 02 / 14 / 04B / 11 / etc.), Year (2025)" \

create_prompt \
  "extrovert" \
  "friendship-maintenance class (uses texting/social media or rarely keeps in touch / maintains long-distance friendships through calls)" \
  "social_connection_frequency (Daily / Weekly / Never / Monthly / weekly / etc.), gathering_size_preference (Small gatherings / Both / Large events / Never), valued_friendship_traits (Support / Loyalty / Honesty / All / All the three and trust / Understanding person / etc.), conflict_resolution_style (Talk it out / Apologize first / Avoid conflict / Let time fix it), family_support_level (Yes / Sometimes / No / Never / yes / etc.), social_battery_recharge (Talking to someone / Alone time / None of the above / Social media), social_orientation_type (Introvert / Ambivert / Extrovert / Not sure), social_anxiety_level (Avoid it / Neutral / Excited / Nervous), emotional_expression_method (Stay silent / Talk openly / Talk openly, Express emotion strongly / Express emotion strongly / Other / etc.), primary_anger_sources (Injustice / Being interrupted / Selfishness / Unnecessary criticism)" \

create_prompt \
  "machine" \
  "machine status (normal operation / at risk of failure)" \
  "Temperature, Vibration, Power_Usage, Humidity, Machine_Type (Mill / Lathe / Drill)" \

create_prompt \
  "postpartum" \
  "postpartum depression status (does not have postpartum depression / has postpartum depression)" \
  "Age, Residence (City / Village), Education Level (University / College / Primary School / High School / No / High school / Primary school / etc.), Marital status (Married / Divorced), Occupation before latest pregnancy (Student / Doctor / Service / Housewife / Teacher / Other / Business / House wife / etc.), Monthly income before latest pregnancy (No / 10000 to 20000 / More than 30000 / Less than 5000 / 5000 to 10000 / 20000 to 30000), Occupation After Your Latest Childbirth (Student / Doctor / Service / Housewife / Teacher / Business / Other / House wife / etc.), Current monthly income (No / 10000 to 20000 / More than 30000 / Less than 5000 / 5000 to 10000 / 20000 to 30000), Husband's education level (University / No / College / Primary School / High School / High school / Primary school / etc.), Husband's monthly income (More than 30000 / No / 10000 to 20000 / 5000 to 10000 / 20000 to 30000 / Less than 5000), Addiction (No / Smoking / Drinking / Drugs), Total children (One / Two / More than two / More than Two / etc.), Disease before pregnancy (No / Non-Chronic Disease / Chronic Disease / 1.2 / etc.), History of pregnancy loss (No / Miscarriage / Still-born Delivery / Still-born delivery / etc.), Family type (Nuclear / Joint), Number of household members (6 to 8 / 2 to 5 / 9 or more), Relationship with the in-laws (Neutral / Good / Bad / Friendly / Poor), Relationship with husband (Good / Neutral / Bad / Friendly / Poor), Relationship with the newborn (Good / Neutral / Bad / Very good), Relationship between father and newborn (Good / Neutral / Bad / Very good), Feeling about motherhood (Neutral / Happy / Sad), Recieved Support (High / Medium / Low), Need for Support (Medium / Low / No / High), Major changes or losses during pregnancy (Yes / No), Abuse (Yes / No), Trust and share feelings (Yes / No), Number of the latest pregnancy (1 / 2 / 3 / 4 / 5 / 7 / 6), Pregnancy length (10 months / 9 months / 8 months / Less than 5 months / 7 months / 6 months), Pregnancy plan (No / Yes), Regular checkups (Yes / No), Fear of pregnancy (Yes / No), Diseases during pregnancy (No / Chronic Disease / Non-Chronic Disease / Non-chronic disease / Chronic disease / etc.), Age of newborn (6 months to 1 year / Older than 1.5 year / 0 to 6 months / 1 year to 1.5 year), Age of immediate older children (No / 13yr or more / 1yr to 3yr / 7yr to 12yr / 4yr to 6yr), Mode of delivery (Normal Delivery / Caesarean Section), Gender of newborn (Boy / Girl), Birth compliancy (No / Yes), Breastfeed (Yes / No), Newborn illness (No / Yes), Worry about newborn (Yes / No), Relax/sleep when newborn is tended (Yes / No), Relax/sleep when the newborn is asleep (Yes / No), Angry after latest child birth (No / Yes), Feeling for regular activities (Worried / Tired / No / Afraid)" \

create_prompt \
  "reading" \
  "reading preference class (reads non-academic books / reads academic books most of the time)" \
  "Gender (Male / Female), Age (18 / 19 / 20 / 21 / 22 / 23), Department (CSE / SWE / English / Pharmacy / EEE / BBA / NFE / Others / MCT / Textile / etc.), Amount of time spent reading books per day, Reason for reading books (Academic Purpose / Keep me informative / Entertainment Purpose / To increase my English vocabulary knowledge / etc.), How to manage books for reading (E-books / Borrow / Others / Buy / Used-all), Number of e-books you read last two years (11-15 / More than 20 / 1-5 / 6-10 / 16-20), Number of printed books you read last two years (11-15 / 1-5 / 6-10 / More than 20 / 16-20), Favorite books for reading to spend leisure time (Biography / Non-Fiction / Others / Fiction / Politics), What is the change that you fell after reading books (Increase Academic Performance / Increase Creative Thinking / Feel Relax / Increase Problem Solving Skill), Who influenced you to read books (Teachers / Self-Decision / Friends / Parents), Which language books do you like to read (English / Bangla / Others), Where do you read printed books (Home / Library / During Travel), Do you read Newspaper (No / Yes), The format that you used for reading books (Both / E-books / Printed), Frequency of reading books (On everyday basis / A few times a week / A few times a month / Less than once a year / Once or twice a year)" \

create_prompt \
  "stars" \
  "star class (dwarf star / giant star)" \
  "Vmag, Plx, e_Plx, B-V, Amag, SpType (G1V / F3V / F7IV / B8/B9V / K5V / F5V / B3III/IV / A4V+... / K3III / B9V / etc.)" \
