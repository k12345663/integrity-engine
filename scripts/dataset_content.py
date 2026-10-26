"""Text of the synthetic submissions (fictional case: VoltRide Mobility).

Relationships are created by *reusing the same Python variables*, so the
ground truth in dataset_manifest.json is exact by construction.
All people, companies and numbers are fictional.
"""

# ---------------------------------------------------------------- template
# The 150-word case prompt every team pastes at the top (Scenario 4).
CASE_PROMPT = (
    "Case Prompt - Inter-University Strategy Challenge 2026, Round 2. VoltRide Mobility operates 1,800 dockless "
    "electric scooters in Pune and completed 410,000 rides in the last financial year. The founders must decide "
    "whether to expand into three tier-2 cities (Nagpur, Indore and Coimbatore) within eighteen months or to deepen "
    "operations in Pune by adding battery-swapping stations and corporate subscriptions. Teams must assess market "
    "attractiveness, unit economics per scooter, regulatory risk and the funding required for each option. Your "
    "submission must recommend one option, justify it with quantitative reasoning, identify the three largest risks "
    "and propose mitigation measures for each. Answers should not exceed 1,500 words excluding this prompt. Use only "
    "the case exhibits and publicly available sources, and cite every external source you rely on. Submissions will "
    "be scored on insight, rigour, feasibility and clarity of communication."
)
SHARED_REFERENCES = [
    "VoltRide Mobility Case Exhibits 1-4, Inter-University Strategy Challenge 2026.",
    "Wikipedia contributors. Scooter-sharing system. Wikipedia, The Free Encyclopedia. Accessed October 2026.",
]

# ---------------------------------------------------------------- sub_01 (source of copies)
A_SITUATION = (
    "VoltRide is not short of demand in Pune. The fleet did 410,000 rides last year, which works out to roughly 228 "
    "rides per scooter, and the app still shows riders waiting at Hinjewadi and Kharadi during the morning peak. The "
    "problem is supply quality, not supply quantity. Exhibit 2 shows that about a fifth of the fleet is unavailable "
    "at any given hour because scooters are either flat or sitting in the depot. Expanding to three new cities would "
    "copy that broken operating model three more times."
)
A_ECONOMICS = (
    "We rebuilt the unit economics from Exhibit 3. A scooter costs 62,000 rupees landed, lasts about thirty months "
    "and earns 212 rupees a day at current utilisation. After charging, van logistics, repairs and insurance it "
    "contributes only 41 rupees a day, so payback takes more than four years, which is longer than the scooter "
    "survives. Any option that does not raise rides per scooter is therefore value destroying, no matter how big the "
    "new market looks on paper."
)
# Paragraph X: read aloud verbatim in pitch_10_sanjay.mp4 (Scenario 1, cross-modal).
A_RECOMMENDATION_X = (
    "Our recommendation is to stay in Pune for the next eighteen months and build a dense battery-swapping network "
    "before chasing new cities. Right now each scooter spends almost forty minutes a day off the road waiting for a "
    "van to collect and recharge it, and that idle time is the biggest reason the fleet earns so little. Twenty swap "
    "stations near the busiest pickup zones would lift utilisation from 5.1 to 6.4 rides per scooter per day."
)
# Paragraphs Y and Z: copied verbatim into sub_02_meera.pdf (exact copy pair).
A_RISK_Y = (
    "The first risk is regulatory. Pune Municipal Corporation has already capped dockless fleets in the old city "
    "wards, and a stricter parking rule could force us to build designated parking bays at our own cost. We would "
    "reduce this exposure by co-funding thirty marked parking hubs with the corporation and sharing anonymised trip "
    "data with its traffic cell, which turns the regulator into a partner rather than an opponent."
)
A_RISK_Z = (
    "The second risk is battery degradation. Swap stations cycle batteries harder than overnight depot charging, and "
    "if pack life falls below eight hundred cycles the station economics break down. A leasing contract with the "
    "battery supplier, priced per kilowatt-hour delivered rather than per pack, moves this risk onto the party that "
    "understands the chemistry best."
)
A_RISK_3 = (
    "The third risk is that a well funded rival enters Pune while we are busy building stations. Honestly we think "
    "this is the smallest of the three, because a swap network is expensive to copy and riders stick with whichever "
    "app has a charged scooter closest to them. Locking in the twenty best station sites early is our main defence."
)

# ---------------------------------------------------------------- sub_02 (copies Y and Z)
B_INTRO = (
    "Team Meridian recommends that VoltRide expands into Indore first and keeps Nagpur and Coimbatore as a second "
    "wave. Indore ranks first in national cleanliness surveys, its central business district is flat, and its "
    "colleges are clustered along a few arterial roads, which makes a compact launch zone possible."
)
B_NUMBERS = (
    "Using the exhibit numbers, a 400 scooter launch in Indore needs about 2.9 crore rupees including the first year "
    "of operating losses. If utilisation reaches the Pune level by month nine, the city breaks even on a cash basis "
    "in month twenty two. We think that is acceptable because the brand becomes regional before competitors arrive."
)
B_CLOSE = (
    "Finally the third risk is execution: hiring a city manager who knows Indore. We suggest seconding two of the "
    "best Pune supervisors for the first six months."
)

# ---------------------------------------------------------------- sub_03 (paraphrase source)
C_PARAS = [
    "Kabir's team argues for a single-city pilot in Indore rather than an immediate three city rollout. The logic is "
    "about learning cheaply. A pilot of 300 scooters costs a fraction of a full launch and tells VoltRide whether its "
    "Pune playbook transfers to a smaller market before serious capital is committed.",
    "Indore suits a pilot for three reasons. The core of the city is flat, so batteries last longer per charge. "
    "Around 1.2 lakh students study within six kilometres of Vijay Nagar, which gives a dense base of short trips. "
    "And no shared scooter operator is active there yet, so early riders will form habits around VoltRide's app.",
    "We would run the pilot through a franchise partner who owns the scooters while VoltRide supplies the software, "
    "the brand and the maintenance standards. This keeps the balance sheet light, and the partner carries the "
    "depreciation risk. Our model shows payback for the partner in fourteen months at five rides per scooter per day.",
    "The decision rule after six months is simple. If rides per scooter exceed 4.5 and the damage rate stays under "
    "three percent, VoltRide signs franchise partners in Nagpur and Coimbatore. If not, it pulls the fleet back to "
    "Pune and has lost less than one crore rupees.",
]
# sub_04: the same arguments, reworded with synonyms and new sentence structure (Scenario 2).
D_PARAS = [
    "In our view VoltRide should test one new market, Indore, instead of entering all three cities at once. The aim "
    "is inexpensive learning: trialling roughly three hundred e-scooters needs only a small share of the money a "
    "complete rollout would, and it reveals whether the operating model built in Pune works in a smaller city "
    "before large investments are made.",
    "There are three reasons Indore is a good test bed. Its central area has almost no slopes, which stretches "
    "battery range on every charge. About 120,000 college students live within a short ride of Vijay Nagar, "
    "producing lots of brief journeys. Finally, no rival sharing company operates in the city today, so the first "
    "users will build their routines around VoltRide.",
    "The trial would be operated by a local franchisee that buys the vehicles, while VoltRide provides the platform, "
    "branding and servicing rules. This approach keeps assets off VoltRide's books and shifts depreciation exposure "
    "to the franchisee, who by our estimates recovers the investment within about fourteen months if each scooter "
    "does five trips daily.",
    "After half a year, a clear threshold decides the next step. Should average trips per vehicle pass 4.5 while "
    "damage remains below 3%, the company recruits franchisees for Nagpur and Coimbatore; otherwise it brings the "
    "scooters home to Pune, having spent under ten million rupees.",
]

# ---------------------------------------------------------------- sub_05 (original, independent)
E_PARAS = [
    "Our team looked at the problem from the customer side. Most VoltRide trips in Pune are under three kilometres "
    "and happen between 8 and 10 in the morning, which means the scooters mostly serve office commuters getting from "
    "a bus stop or metro station to their desk.",
    "That pattern points to corporate subscriptions. If an IT park like Magarpatta buys monthly passes for its "
    "employees, VoltRide gets predictable revenue and can park scooters exactly where the demand is. We priced a "
    "pass at 499 rupees per employee per month for twenty rides, roughly a third cheaper than pay per ride.",
    "We spoke informally to four HR managers while preparing this. Three of them said they already reimburse cab "
    "fares for late shifts and would happily move some of that budget to scooters if the company handled insurance "
    "and helmets. That's a small sample, but it suggests the willingness to pay is real.",
    "Our main worry is that subscriptions cannibalise existing riders who already pay full price. To limit that, "
    "passes would only be sold to employers, not to individuals, and would only unlock scooters inside geofenced "
    "office zones during working hours.",
]

# ---------------------------------------------------------------- sub_06 (copies from Wikipedia)
# Two passages lifted from the public Wikipedia article "Scooter-sharing system" (CC BY-SA 4.0).
WIKI_COPY_1 = (
    "A scooter-sharing system or kicksharing system is a shared transport service in which e-scooters are made "
    'available to use for short-term rentals. E-scooters are typically "dockless," meaning that they do not have a '
    "fixed home location and are dropped off and picked up from certain locations in the service area."
)
WIKI_COPY_2 = (
    "To rent a dockless e-scooter, users download a smartphone application. The application shows users a map of "
    "nearby e-scooters and enables them to unlock them. Scooters are equipped with built-in GPS chips and cellular "
    "connectivity which allows them to report their location in real time during a trip."
)
F_OWN = [
    "VoltRide should think carefully before spreading itself across three cities. Our analysis focuses on Nagpur, "
    "which has wide roads but extreme summer heat, and we worry that battery packs will degrade quickly when "
    "afternoon temperatures stay above forty five degrees for weeks.",
    "We would start Nagpur with a smaller fleet of 250 scooters and run them only in the mornings and evenings "
    "during May and June. That limits revenue but protects the most expensive part of the vehicle.",
]

# ---------------------------------------------------------------- sub_07 & sub_15 (LLM-style text)
# Written in the default style of a chat assistant asked to "write a case competition answer".
G_AI = [
    "In today's rapidly evolving urban mobility landscape, VoltRide Mobility stands at a pivotal crossroads. The "
    "decision between expanding into tier-2 cities and deepening its presence in Pune is multifaceted, and it is "
    "important to note that each option carries distinct opportunities and challenges.",
    "Furthermore, a comprehensive evaluation of market attractiveness reveals that Nagpur, Indore and Coimbatore "
    "offer significant untapped potential. These cities are experiencing robust economic growth, rising disposable "
    "incomes and a growing appetite for sustainable transportation solutions. By leveraging its proven operational "
    "expertise, VoltRide can seamlessly replicate its success and foster brand loyalty across new markets.",
    "Moreover, unit economics play a crucial role in this decision. Expansion allows VoltRide to harness economies "
    "of scale in procurement, maintenance and technology, thereby optimizing cost structures and enhancing overall "
    "profitability. A holistic approach that integrates data-driven insights with local partnerships will be "
    "paramount to navigating the complexities of new regulatory environments.",
    "However, it is crucial to acknowledge the associated risks. Regulatory uncertainty, operational complexity and "
    "competitive pressure could undermine the expansion strategy. To mitigate these risks, VoltRide should foster "
    "strong relationships with municipal authorities, invest in robust operational frameworks and differentiate "
    "itself through superior customer experience.",
    "In conclusion, expanding into tier-2 cities represents a strategic imperative for VoltRide. By embracing "
    "innovation, leveraging strategic partnerships and maintaining a steadfast commitment to sustainability, "
    "VoltRide is well positioned to unlock new avenues of growth and solidify its position as a leader in India's "
    "ever-evolving micromobility ecosystem.",
]
O_AI = [
    "Executive Summary. In an ever-evolving micromobility landscape, VoltRide must carefully navigate the trade-off "
    "between geographic expansion and operational depth. This analysis delves into both strategic pathways and "
    "provides a comprehensive, data-driven recommendation.",
    "Deepening operations in Pune offers a compelling value proposition. Battery-swapping infrastructure can "
    "significantly enhance fleet availability, while corporate subscriptions foster predictable recurring revenue. "
    "Moreover, concentrating resources allows VoltRide to strengthen its competitive moat and cultivate a loyal, "
    "engaged customer base.",
    "Conversely, expansion into tier-2 cities unlocks substantial growth potential. Nevertheless, it introduces "
    "multifaceted challenges, including regulatory complexity, capital intensity and the need to build local "
    "operational capabilities from the ground up. It is worth noting that these challenges are not "
    "insurmountable, but they require careful planning and robust execution.",
    "Ultimately, a phased approach strikes the optimal balance. VoltRide should first strengthen its Pune "
    "foundation, leveraging swapping stations to improve unit economics, and subsequently pursue a measured "
    "expansion. This holistic strategy underscores the importance of sustainable growth and positions VoltRide for "
    "long-term success in a dynamic and competitive market.",
    "In summary, by harnessing operational excellence and strategic foresight, VoltRide can seamlessly balance "
    "short-term profitability with long-term ambition, fostering resilient growth across India's vibrant "
    "urban mobility ecosystem.",
]

# ---------------------------------------------------------------- sub_08 (heavily formatted, original)
H_COL_LEFT = [
    "Coimbatore is the option nobody in our class picked, which is exactly why we looked at it. The city has a large "
    "textile and engineering workforce that commutes short distances between residential colonies and factory "
    "clusters, and the weather is mild most of the year.",
    "Our survey of 63 factory workers in Peelamedu found that 41 currently walk more than fifteen minutes to the "
    "nearest bus stop. A shift-timed scooter service could cut that walk to four minutes.",
]
H_COL_RIGHT = [
    "The catch is price sensitivity. Workers told us they would pay at most fifteen rupees a trip, about half of "
    "what VoltRide charges in Pune. That only works if employers subsidise the rides as a welfare benefit.",
    "We therefore suggest a pilot co-funded by two large mills, with fixed morning and evening shift windows, and a "
    "review after ninety days.",
]
H_TABLE = [
    ["Metric", "Pune today", "Coimbatore pilot"],
    ["Scooters", "1,800", "200"],
    ["Rides per scooter per day", "5.1", "6.0 (shift peaks)"],
    ["Price per ride (Rs)", "30", "15 + 15 employer subsidy"],
]

# ---------------------------------------------------------------- sub_09 (original, independent)
I_PARAS = [
    "Before choosing between growth options, we think VoltRide has to fix rider safety. The case exhibits mention 37 "
    "reported injuries last year but say nothing about helmets, and we couldn't find a single photo in VoltRide's "
    "marketing where a rider wears one.",
    "Insurers are starting to notice. A national insurer quoted us an indicative premium that would rise by about "
    "sixty percent per scooter if the injury rate stays where it is. That increase alone wipes out most of the margin "
    "gained from either growth plan.",
    "Our proposal is a helmet-lock: each scooter carries a foldable helmet in a smart box, and the ride cannot start "
    "until the helmet sensor is clipped. A pilot in Kothrud would show whether riders tolerate the extra thirty "
    "seconds at unlock.",
    "Only after injury rates fall below one per ten thousand rides would we revisit the expansion question. A "
    "company that hurts its customers will struggle to win permits in any new city anyway.",
]

# ---------------------------------------------------------------- video scripts
PITCH_10_INTRO = "Hi everyone, I'm Sanjay from team Orbit, and here is our pitch for VoltRide."
PITCH_10_OUTRO = "Thank you for listening."
PITCH_11 = (
    "Hello judges, this is Leela from team Northstar. We believe VoltRide should launch in Coimbatore with women "
    "commuters as the first customer segment. Our interviews showed that many women avoid crowded buses in the "
    "evening, and a scooter with a panic button and live trip sharing would feel safer. We would start with one "
    "hundred and fifty scooters near the women's colleges. Thank you."
)
