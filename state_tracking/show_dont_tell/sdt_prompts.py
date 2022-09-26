# Copyright 2021 Google Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompts for Show Don't Tell models."""


import collections

Prompt = collections.namedtuple('Prompt', ['utt', 'slots'])
OrderedDict = collections.OrderedDict

SGD_SEPARATED_ANNOTATION_PROMPTS = {
    'Alarm_1': [
        Prompt(
            utt='[user] What alarms do I have please? [system] You have 1 alarm set for 7 am that is called Wake up. [user] Ok, perfect! [system] Do you want to add any alarms? [user] Yes, I would like to add one for 5:15 in the evening. [system] Please confirm that you are creating an alarm at 5:15 pm that is called New alarm. [user] Yes, that is correct.',
            slots=OrderedDict([('new_alarm_time', '5:15 in the evening'),
                               ('new_alarm_name', 'New alarm')])),
        Prompt(
            utt="[user] I need to set an alarm. [system] Sure, I can do that. What time? [user] Set it for 15:15. [system] Please confirm that you'd like a new alarm called new alarm set at 3:15 pm. [user] Yup, thanks!",
            slots=OrderedDict([('new_alarm_name', 'new alarm'),
                               ('new_alarm_time', '15:15')])),
        Prompt(
            utt='[user] When are my alarms? [system] You have 2 alarms set. One is 6:30 am called wake up. [user] Ok, I need to add another at quarter to 4 in the afternoon. [system] Please confirm: add an alarm for 3:45 pm titled new alarm [user] Yes.',
            slots=OrderedDict([('new_alarm_name', 'new alarm'),
                               ('new_alarm_time', '3:45 pm')])),
        Prompt(
            utt='[user] Please show me my alarms. [system] You have a Wake up alarm at 6:30 am. [user] Thanks, that is great. [system] Would you like me to add an alarm? [user] Yes, please add an alarm for quarter to 5 in the evening called Phone home.',
            slots=OrderedDict([('new_alarm_name', 'Phone home'),
                               ('new_alarm_time', 'quarter to 5 in the evening')
                              ])),
        Prompt(
            utt="[user] Yo, I'd like to set an alarm for 5 pm. [system] Let me see if I got everything correctly: You'd like a New alarm created for 5 pm. Correct? [user] That is correct, just call the alarm grocery run.",
            slots=OrderedDict([('new_alarm_name', 'grocery run'),
                               ('new_alarm_time', '5 pm')]))
    ],
    'Banks_1': [
        Prompt(
            utt="[user] What's my balance? [system] In checking or savings? [user] In checking. [system] Your checking account has $5,118.77. [user] Ok, I want to transfer some money. [system] To whom? [user] To Amir. [system] And what amount would you like to transfer? [user] Send $1,630. [system] Please confirm: Transfer $1,630 from your checking account to Amir's checking account. [user] Confirmed.",
            slots=OrderedDict([
                ('account_type',
                 'checking of possible values checking, savings'),
                ('recipient_account_name', 'Amir'), ('amount', '$1,630'),
                ('recipient_account_type',
                 'checking of possible values checking, savings')
            ])),
        Prompt(
            utt='[user] I need to make a transfer for 1770 dollars to Xiaoxue. [system] You want me to transfer $1,770 to Xiaoxue checking account from your checking account. [user] Yes I do.',
            slots=OrderedDict([
                ('account_type',
                 'checking of possible values checking, savings'),
                ('amount', '$1,770'), ('recipient_account_name', 'Xiaoxue'),
                ('recipient_account_type',
                 'checking of possible values checking, savings')
            ])),
        Prompt(
            utt='[user] I wanna make a transfer [system] what account from? [user] the checking account [system] where to send money? [user] send to sanuj [system] what amount then? [user] $1,240 is good [system] confirm checking $1,240 to sanuj checking [user] yes that works',
            slots=OrderedDict([
                ('account_type',
                 'checking of possible values checking, savings'),
                ('amount', '$1,240'), ('recipient_account_name', 'sanuj'),
                ('recipient_account_type',
                 'checking of possible values checking, savings')
            ])),
        Prompt(
            utt="[user] I want to check my savings account balance. [system] Your savings account balance is $5,250.80. [user] Ok. I want to make a transfer of 320 bucks. Send it to Lily. [system] Please confirm the following details: Transfer 320 bucks from your savings account to Lily's checking account. [user] Yes.",
            slots=OrderedDict([('account_type',
                                'savings of possible values checking, savings'),
                               ('amount', '320 bucks'),
                               ('recipient_account_name', 'Lily'),
                               ('recipient_account_type',
                                'checking of possible values checking, savings')
                              ])),
        Prompt(
            utt="[user] hey can you send over $380? [system] who do you wanna send to? [user] give it to ralph [system] use yourchecking or savings? [user] checking works for this [system] $380 from your checking account to ralph's checking, that alright? [user] no, his savings please",
            slots=OrderedDict([
                ('account_type',
                 'checking of possible values checking, savings'),
                ('amount', '$380'), ('recipient_account_name', 'ralph'),
                ('recipient_account_type',
                 'savings of possible values checking, savings')
            ]))
    ],
    'Banks_2': [
        Prompt(
            utt="[user] What's my balance? [system] In checking or savings? [user] In savings. [system] You have $5,984.42 in savings. [user] Ok, I want to transfer to someone's savings. [system] To whom? [user] To Diego. [system] Amount of transfer? [user] Send $1,210. [system] Please confirm: transfer $1,210 from savings to Diego in savings. [user] Yeah.",
            slots=OrderedDict([('account_type',
                                'savings of possible values checking, savings'),
                               ('recipient_account_type',
                                'savings of possible values checking, savings'),
                               ('recipient_name', 'Diego'),
                               ('transfer_amount', '$1,210')])),
        Prompt(
            utt="[user] hey can you open my banking app? [system] wanna make a transfer? [user] yes I do [system] who will you send to? [user] three hundred and twenty dollars to mahmoud [system] $320 from your checking to mahmoud's checking, alright? [user] yes that works, when will he get the money?",
            slots=OrderedDict([
                ('account_type',
                 'checking of possible values checking, savings'),
                ('recipient_account_type',
                 'checking of possible values checking, savings'),
                ('recipient_name', 'mahmoud'), ('transfer_amount', '$320')
            ])),
        Prompt(
            utt="what is my checking balance please? [system] your checking has $10,960.26 right now [user] okay thank you, send one thousand three hundred and forty bucks from my checking to grace's savings, thank you",
            slots=OrderedDict([
                ('account_type',
                 'checking of possible values checking, savings'),
                ('recipient_account_type',
                 'savings of possible values checking, savings'),
                ('recipient_name', 'grace'),
                ('transfer_amount',
                 'one thousand three hundred and forty bucks')
            ])),
        Prompt(
            utt="[user] Will you check my savings balance? [system] Balance is $5,567.84 in your savings. [user] Great. [system] Do you want to make a transfer? [user] Yes. [system] How much and who do you want to send money to? [user] Transfer $1,030 to Justin's checking",
            slots=OrderedDict([
                ('account_type',
                 'savings of possible values checking, savings'),
                ('recipient_account_type',
                 'checking of possible values checking, savings'),
                ('recipient_name', 'Justin'), ('transfer_amount', '$1,030')
            ])),
        Prompt(
            utt="[user] Can you check my balance please [system] Sure. Checking or savings? [user] Let's do checking first [system] Your checking balance is $23,352.71 [user] OK, how about savings? [system] That'd be $7,976.46 [user] Aight, please transfer 1,790 bucks to mom's savings account",
            slots=OrderedDict([
                ('account_type',
                 'savings of possible values checking, savings'),
                ('recipient_account_type',
                 'checking of possible values checking, savings'),
                ('recipient_name', 'mom'), ('transfer_amount', '1,790 bucks')
            ]))
    ],
    'Buses_1': [
        Prompt(
            utt='[user] Can you find me a bus to LAX for 4 people? [system] What is the departure location and time? [user] The 1st of March from San Francisco. [system] I found 9 buses for you. One departs at 8:40 am and costs $41. It has 0 transfers. [user] That is perfect.',
            slots=OrderedDict([('to_location', 'LAX'),
                               ('from_location', 'San Francisco'),
                               ('leaving_date', '1st of March'),
                               ('travelers',
                                '4 of possible values 1, 2, 3, 4, 5'),
                               ('leaving_time', '8:40 am')])),
        Prompt(
            utt="[user] I need to buy a bus ticket from San Diego. Just one person [system] Sure. Where to? What day do you wan to travel [user] Headed to San Francisco on the 10th of this month [system] There's a 0 transfer bus that departs at 7:40 am, it costs $52 [user] Oh that's nice",
            slots=OrderedDict([('from_location', 'San Diego'),
                               ('leaving_date', '10th of this month'),
                               ('leaving_time', '7:40 am'),
                               ('to_location', 'San Francisco'),
                               ('travelers',
                                '1 of possible values 1, 2, 3, 4, 5')])),
        Prompt(
            utt="[user] I need to book a bus ticket, can you find me a seat? [system] Sure. Where are you going? When are you leaving? [user] I'd like to get four tickets from San Fran to Sacramento, CA on the 12th of March. [system] Okay, the bus leaves at 7:20 am and costs $22 per person, it has 0 transfers. [user] That's great.",
            slots=OrderedDict([('from_location', 'San Fran'),
                               ('leaving_date', '12th of March'),
                               ('leaving_time', '7:20 am'),
                               ('to_location', 'Sacramento, CA'),
                               ('travelers',
                                '4 of possible values 1, 2, 3, 4, 5')])),
        Prompt(
            utt='[user] find me a bus leaving 8th of this month [system] where to then? [user] going to san francisco [system] where are you leaving from? [user] portland, and to seattle instead [system] found one at 8:40 am for $30, no transfers [user] get me one seat on that please bye now',
            slots=OrderedDict([('from_location', 'portland'),
                               ('leaving_date', '8th of this month'),
                               ('leaving_time', '8:40 am'),
                               ('to_location', 'seattle'),
                               ('travelers',
                                '1 of possible values 1, 2, 3, 4, 5')])),
        Prompt(
            utt='[user] Hello, help me look up a bus please [system] what is your origin and destination? [user] las vegas to fresno on the 9th of march [system] I found 10 buses. The first direct one leaves at 7:10 am costing $45 [user] that is good [system] wanna get tickets? [user] yes for two',
            slots=OrderedDict([('from_location', 'las vegas'),
                               ('leaving_date', '9th of march'),
                               ('leaving_time', '7:10 am'),
                               ('to_location', 'fresno'),
                               ('travelers',
                                '2 of possible values 1, 2, 3, 4, 5')]))
    ],
    'Buses_2': [
        Prompt(
            utt='[user] I need help finding a bus to LAX. [system] Where would you be departing from? [user] I will be leaving from Vegas. [system] What day do you plan on leaving? [user] My departure date is the 3rd of this month. I need 3 tickets and prefer Economy tickets. [system] I have a bus that leaves at 11:50 am that has Economy tickets for $33 each. [user] That one sounds perfect.',
            slots=OrderedDict([
                ('destination', 'LAX'), ('origin', 'Vegas'),
                ('departure_date', '3rd of this month'),
                ('fare_type',
                 'Economy of possible values Economy, Economy extra, Flexible'),
                ('group_size', '3 of possible values 1, 2, 3, 4, 5'),
                ('departure_time', '11:50 am')
            ])),
        Prompt(
            utt="[user] I'm looking for a bus and want to use Flexible tickets. [system] Can you name the place of departure. [user] This will be for three people leaving SD. [system] What is your destination and when would you like to leave? [user] Fresno, CA and the 4th of this month. [system] There are 6 buses, this bus leaves at 8:50 am and is a Flexible ticket for $63. [user] Awesome.",
            slots=OrderedDict([
                ('departure_date', '4th of this month'),
                ('departure_time', '8:50 am'), ('destination', 'Fresno, CA'),
                ('fare_type',
                 'Flexible of possible values Economy, Economy extra, Flexible'
                ), ('group_size', '3 of possible values 1, 2, 3, 4, 5'),
                ('origin', 'SD')
            ])),
        Prompt(
            utt='[user] Will you help me find a bus that leaves from SD? [system] When and where are you going? [user] I am going to Anaheim, CA next Thursday. [system] I found 9 buses. One of them leaves at 10:10 am and costs $25 on Economy. [user] Anything else? I need tickets for three people. Any type is okay. [system] I found one bus that leaves at 10:40 am. [user] Sounds good',
            slots=OrderedDict([
                ('departure_date', 'next Thursday'),
                ('departure_time', '10:40 am'), ('destination', 'Anaheim, CA'),
                ('fare_type',
                 'Economy of possible values Economy, Economy extra, Flexible'),
                ('group_size', '3 of possible values 1, 2, 3, 4, 5'),
                ('origin', 'SD')
            ])),
        Prompt(
            utt="[user] I'm planning a trip, can you find me a bus ticket? [system] Sure, where from? [user] From Portland and going to Vancouver. I'll need 2 tickets [system] No problem. When do you want to leave? [user] On the 5th of this month please [system] I've found 8, there's an economy ticket for $39 that leaves at 6:50 am [user] Yeah that sounds great [system] Do you want me to book them for you? [user] No thanks, I'll book them later.",
            slots=OrderedDict([
                ('departure_date', '5th of this month'),
                ('departure_time', '6:50 am'), ('destination', 'Vancouver'),
                ('fare_type',
                 'Economy of possible values Economy, Economy extra, Flexible'),
                ('group_size', '2 of possible values 1, 2, 3, 4, 5'),
                ('origin', 'Portland')
            ])),
        Prompt(
            utt='[user] I Need a bus now. Can you help? [system] Your destination please. [user] It is to SD. [system] Can you provide time and place of embarking? [user] It is on Wednesday next week from Long Beach. [system] Ok, there is one at 7 am with Economy ticket at $24. [user] There are three of us. Are there any other buses. [system] Another leaves at 8:40 am with economy ticket of $25. [user] That is fine.',
            slots=OrderedDict([
                ('departure_date', 'Wednesday next week'),
                ('departure_time', '8:40 am'), ('destination', 'SD'),
                ('fare_type',
                 'Economy of possible values Economy, Economy extra, Flexible'),
                ('group_size', '3 of possible values 1, 2, 3, 4, 5'),
                ('origin', 'Long Beach')
            ]))
    ],
    'Buses_3': [
        Prompt(
            utt="[user] I need help looking for a bus. It doesn't have to be direct. [system] What date will you be travelling on and where are you travelling to and from. [user] I want to travel to SD from Fresno, CA on the 11th of March [system] There's one that leaves at 8:20 am for $48. [user] Okay, that works. Can you book me one ticket? [system] I'll purchase you 1 ticket for the 8:20 am bus from Fresno to San Diego on March 11th, no extra luggage. Is that what you want? [user] Yeah, that's it.",
            slots=OrderedDict([
                ('category', 'dontcare of possible values direct, one-stop'),
                ('departure_date', '11th of March'), ('from_city', 'Fresno'),
                ('to_city', 'SD'), ('departure_time', '8:20 am'),
                ('num_passengers', '1 of possible values 1, 2, 3, 4, 5'),
                ('additional_luggage', 'False of possible values True, False')
            ])),
        Prompt(
            utt="[user] Get me a bus, leaving from SF to Long Beach, a direct one at that. [system] When are you leaving? [user] Leaving Thursday next week [system] There are 7 buses and there is one departing at 7 am and costs $48 [user] That is great [system] Should i get the tickets? [user] Yes, please do [system] how many tickets? [user] Make it for four people. We don't have bags to check in.",
            slots=OrderedDict([
                ('additional_luggage', 'False of possible values True, False'),
                ('category', 'direct of possible values direct, one-stop'),
                ('departure_date', 'Thursday next week'),
                ('departure_time', '7 am'), ('from_city', 'SF'),
                ('num_passengers', '4 of possible values 1, 2, 3, 4, 5'),
                ('to_city', 'Long Beach')
            ])),
        Prompt(
            utt="[user] Would you find me a direct bus? [system] When will you leave? [user] I'll leave on March 4th. [system] Where will you leave from and where are you going? [user] I will leave alone from Anaheim and go to LAX. [system] How about a bus leaving at 10:20 am that costs $16? [user] Okay, purchase tickets just for me. I have a bag to check in, so add that too.",
            slots=OrderedDict([
                ('additional_luggage', 'True of possible values True, False'),
                ('category', 'direct of possible values direct, one-stop'),
                ('departure_date', 'March 4th'), ('departure_time', '10:20 am'),
                ('from_city', 'Anaheim'),
                ('num_passengers', '1 of possible values 1, 2, 3, 4, 5'),
                ('to_city', 'LAX')
            ])),
        Prompt(
            utt="[user] I'm going on a short trip soon. Can you help me find a bus ride with one-stop to Anaheim? [system] Sure, when do you want to leave and where from? [user] I'd be leaving from Sacramento on the 11th of this month. Oh, and change the destination to Long Beach. [system] Ok, there's a bus at 7:40 am for $58. How about that? [user] Yeah, that sounds fine. [system] Ok, anything else I can help with? [user] Yes, can you go ahead and book two tickets for me? [system] Sure, any bags to check in? [user] Naah that's okay.",
            slots=OrderedDict([
                ('additional_luggage', 'False of possible values True, False'),
                ('category', 'one-stop of possible values direct, one-stop'),
                ('departure_date', '11th of this month'),
                ('departure_time', '7:40 am'), ('from_city', 'Sacramento'),
                ('num_passengers', '2 of possible values 1, 2, 3, 4, 5'),
                ('to_city', 'Long Beach')
            ])),
        Prompt(
            utt="[user] I need a direct bus on the 1st. [system] Where do you want to go and from which place? [user] I am moving from Las Vegas to San Diego. [system] I searched and found 1 bus for you. What is your opinion about the bus starting at 10:50 am. The cost is $41. [user] It is good. [system] Shall I get the tickets? [user] Yes Please book 4 seats. We'll also have a large bag.",
            slots=OrderedDict([
                ('additional_luggage', 'True of possible values True, False'),
                ('category', 'direct of possible values direct, one-stop'),
                ('departure_date', 'the 1st'), ('departure_time', '10:50 am'),
                ('from_city', 'Las Vegas'),
                ('num_passengers', '4 of possible values 1, 2, 3, 4, 5'),
                ('to_city', 'San Diego')
            ]))
    ],
    'Calendar_1': [
        Prompt(
            utt="[user] Can you check my schedule on March 13th? [system] On March 13th, your schedule shows an open period of time between 12:30 pm and 2 pm. [user] That's great. Thanks. [system] Did you need to add an event to the calendar? [user] Yes, Can you please add the Chris Webby concert? The event is being held at 2367 Shattuck Avenue. [system] What time does it begin? [user] It starts at 12:30pm.",
            slots=OrderedDict([('event_date', 'March 13th'),
                               ('event_location', '2367 Shattuck Avenue'),
                               ('event_name', 'Chris Webby concert'),
                               ('event_time', '12:30 pm')])),
        Prompt(
            utt="[user] Hey what is aegena apartment's address? [system] 1290 san tomas aquino road [user] okay, I have an apartment visit there 3 o'clock in the afternoon march 6th, add that to my calendar pretty please? [system] Adding event apartment visit on march 6th 3pm at 1290 san tomas aquino road [user] yes that works",
            slots=OrderedDict([('event_date', 'march 6th'),
                               ('event_location', '1290 san tomas aquino road'),
                               ('event_name', 'apartment visit'),
                               ('event_time', "3 o'clock in the afternoon")])),
        Prompt(
            utt='[user] can you add my stylist appointment to my calendar? march 2nd [system] Sure, event name stylist works? Also, what time? [user] Yep, 6pm [system] Should I add an address? [user] Oh yes 3136 contra loma boulevard please',
            slots=OrderedDict([('event_date', 'march 2nd'),
                               ('event_location', '3136 contra loma boulevard'),
                               ('event_name', 'stylist'),
                               ('event_time', '6 pm')])),
        Prompt(
            utt="[user] Am I available Saturday this week? [system] There are 3 empty slots on your calendar. From 4 pm to 8:30 pm you have time [user] Sounds good, add an event to my calendar [system] What event is it, and when and where? [user] It's a hairdresser appointment at 5 in the evening at 994 The Alameda [system] Confirming your 5 pm Hairdresser appointment Saturday this week at 994 The Alameda [user] Yes thats right",
            slots=OrderedDict([('event_date', 'Saturday this week'),
                               ('event_location', '994 The Alameda'),
                               ('event_name', 'Hairdresser appointment'),
                               ('event_time', '5 in the evening')])),
        Prompt(
            utt="[user] Add an appointment on my calendar for 15:30. [system] Where are you going for this appointment? [user] I'm going to a Property viewing at 100 Summerfield Drive. [system] So do I have this right? You have a Property viewing at 100 Summerfield Drive on March 14th at 3:30 pm? [user] Yes, that's correct.",
            slots=OrderedDict([('event_date', 'March 14th'),
                               ('event_location', '100 Summerfield Drive'),
                               ('event_name', 'Property viewing'),
                               ('event_time', '15:30')]))
    ],
    'Events_1': [
        Prompt(
            utt="[user] Anything interesting happening in Toronto? [system] What do you have in mind? Music? Sports? [user] Check out some Sports. Say, how about a Baseball game on the 1st of March. [system] Well, today we have the Blue Jays Vs Braves at Rogers Centre. First pitch is 8 pm. [user] Cool. I'm down with that. [system] Should I pick you up some tickets? [user] Yeah, please do. [system] How many? [user] Just one.",
            slots=OrderedDict([
                ('city_of_event', 'Toronto'),
                ('category', 'Sports of possible values Music, Sports'),
                ('date', '1st of March'), ('subcategory', 'Baseball'),
                ('event_name', 'Blue Jays Vs Braves'),
                ('number_of_seats',
                 '1 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt="[user] I'd like to watch a sport, maybe baseball [system] There's mets vs braves at the citi field march 13th at 6 pm [user] that is good [system] wanna get tickets? This is in new york [user] Yes please, I would like two",
            slots=OrderedDict([
                ('category', 'Sports of possible values Music, Sports'),
                ('city_of_event', 'new york'), ('date', 'march 13th'),
                ('event_name', 'mets vs braves'),
                ('number_of_seats',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('subcategory', 'baseball')
            ])),
        Prompt(
            utt="[user] I'd love a blues music concert on the the 2nd [system] which city do you want? [user] I want something in sd [system] zz top is performing at the cal coast credit union amphitheater. That's on march 2nd at 6 pm [user] tell me the address of the venue [system] 5500 campanile drive [user] Great, get me 3 tix",
            slots=OrderedDict([
                ('category', 'Music of possible values Music, Sports'),
                ('city_of_event', 'sd'), ('date', 'march 2nd'),
                ('event_name', 'zz top'),
                ('number_of_seats',
                 '3 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('subcategory', 'blues')
            ])),
        Prompt(
            utt='[user] Can you find a sports game, say baseball, to watch march 2nd in new york city? [system] I found 10. How about mets vs cubs at the citi field starting 7:30 pm [user] that is good. I want tickets to this game please. [system] how many then? [user] for me and my husband.',
            slots=OrderedDict([
                ('category', 'Sports of possible values Music, Sports'),
                ('city_of_event', 'new york'), ('date', 'march 2nd'),
                ('event_name', 'mets vs cubs'),
                ('number_of_seats',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('subcategory', 'baseball')
            ])),
        Prompt(
            utt="[user] I'm looking for something stimulating to do. [system] Where? [user] San Fran. [system] Any event type preference? [user] Pop Music on the 7th of March. [system] The Chapel is featuring Remo Drive March 7th at 6 pm. [user] That's perfect. [system] Would you like me to book seats. [user] Yes, 3 tickets. [system] Enjoy the day.",
            slots=OrderedDict([
                ('category', 'Music of possible values Music, Sports'),
                ('city_of_event', 'San Fran'), ('date', '7th of March'),
                ('event_name', 'Remo Drive'),
                ('number_of_seats',
                 '3 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('subcategory', 'Pop')
            ]))
    ],
    'Events_2': [
        Prompt(
            utt='[user] I would like to go for event could you help me to find something interesting? [system] What are you interested in looking for? Do you have any preference on the city to look for the event? [user] I am looking for sports events like Football. What about the events in Philly. [system] The Event is Eagles Vs Cowboys and it is at Lincoln Financial Field. [user] That sound perfect for me. I would like to buy four tickets on March 10th for that event... oh no just one ticket please.',
            slots=OrderedDict([
                ('category', 'Football'), ('city', 'Philadelphia'),
                ('event_type', 'Sports of possible values Music, Sports'),
                ('event_name', 'Eagles Vs Cowboys'), ('date', 'March 10th'),
                ('number_of_tickets',
                 '1 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt="[user] I'd like to find pop music events. [system] What city are you interested in? [user] In Philadelphia please. [system] I have a Conan Gray concert at the Fillmore Philadelphia. [user] I like that one. What is the date? [system] That's happening next Monday. [user] I'll take that. [system] Would you like to buy tickets? [user] I would, 2 please",
            slots=OrderedDict([
                ('category', 'pop'), ('city', 'Philadelphia'),
                ('date', 'next Monday'), ('event_name', 'Conan Gray'),
                ('event_type', 'Music of possible values Music, Sports'),
                ('number_of_tickets',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt="[user] Please check the weather in LAX for March 10th. [system] The average temperature for the day is forecast at 91 degrees, with a 10 percent chance of rain. [user] Okay. I will need something fun to do. Are there any concert events? I prefer country. [system] I have 1 event you may like. How about Bill Callahan at the Lodge Room? [user] That sounds good to me. Let's buy tickets. [system] How many tickets do you want? [user] Just one ticket, please. [system] Please confirm: I am booking 1 ticket for Bill Callahan on March 10th in Los Angeles. [user] Yes, that's perfect. [system] Your reservation has been made. Have a great time! [user] When does the event start? [system] It starts at 6:30 pm. [user] Thanks, that's all I needed. [system] Have a good day.",
            slots=OrderedDict([
                ('category', 'country'), ('city', 'LAX'),
                ('date', 'March 10th'), ('event_name', 'Bill Callahan'),
                ('event_type', 'Music of possible values Music, Sports'),
                ('number_of_tickets',
                 '1 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt="[user] I'd like to search for events. [system] What are you interested in? [user] I'm looking for a concert. [system] Do you have preference for which city? [user] I'm looking for Hip Hop concerts in NYC. [system] I've found 4. One is Blackbear at Pier 17. [user] Where is it located and at what time does it start? [system] It starts at 89 South Street, 6:30 pm. [user] Sounds good to me. [system] Would you like to buy a ticket? [user] Yes. I'd like a ticket for the 13th of March. [system] How many tickets would you like? [user] I'd like 2 tickets. [system] Please confirm the following: You'd like 2 tickets for Blackbear on March 13th in New York. [user] Yes, that works for me. [system] Great, your reservation has been made. [user] Thanks for your help. That's all I need. [system] Have a great day.",
            slots=OrderedDict([
                ('category', 'Hip Hop'), ('city', 'NYC'),
                ('date', '13th of March'), ('event_name', 'Blackbear'),
                ('event_type', 'Music of possible values Music, Sports'),
                ('number_of_tickets',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt='[user] I am searching for Concert events on the 12th of March. [system] Do you have any city preferences? [user] I prefer events in New York. I like Blues events. [system] I found 5 events which may suit your taste. There is a Ben Harper event held at Pier 17. [user] That sounds good to me. I want to purchase 3 tickets. [system] Please confirm these details: 3 tickets for Ben Harper event on March 12th in New York. [user] Yes, that works for me. [system] Your reservation has been made. [user] Where is the venue located? [system] The venue is located at 89 South Street. [user] When does the event start? [system] The event starts at 6 pm. [user] Thanks a lot for your help. [system] Is there anything else I can help you with? [user] No. Thank you for your help. [system] Have a great day.',
            slots=OrderedDict([
                ('category', 'Blues'), ('city', 'New York'),
                ('date', '12th of March'), ('event_name', 'Ben Harper'),
                ('event_type', 'Music of possible values Music, Sports'),
                ('number_of_tickets',
                 '3 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ]))
    ],
    'Events_3': [
        Prompt(
            utt='[user] I am in search of some interesting show around Portland. Are there? [system] There are 10 events. At Newmark Theatre a popular event, Annual School Performance is taking place on March 2nd. Do you like? [user] Yeah, please get one ticket.',
            slots=OrderedDict([
                ('city', 'Portland'),
                ('event_type', 'Theater of possible values Music, Theater'),
                ('date', 'March 2nd'),
                ('event_name', 'Annual School Performance'),
                ('number_of_tickets',
                 '1 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt="[user] I'm in the mood to see a play, can you find me one? [system] In what city? [user] I'm looking in the San Diego area. [system] I found 10 plays. About Time is popular and is at Talent to aMuse Theatre Co March 9th at 4 pm. [user] Perfect. I'd like to book tickets for myself plus a friend.",
            slots=OrderedDict([
                ('city', 'San Diego'), ('date', 'March 9th'),
                ('event_name', 'About Time'),
                ('event_type', 'Theater of possible values Music, Theater'),
                ('number_of_tickets',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt="[user] CAn you help me find a musical performance that I can catch on March 10th in Yerevan? [system] Sure! How about seeing Tigran Hamasyan at the Opera House on March 10th at 6 pm [user] That's great, now find me 1 hotel room in that area too [system] I'll look that up shortly, but did you want tickets to the concert? [user] Oh right, get me 3 tickets to that show",
            slots=OrderedDict([
                ('city', 'Yerevan'), ('date', 'March 10th'),
                ('event_name', 'Tigran Hamasyan'),
                ('event_type', 'Music of possible values Music, Theater'),
                ('number_of_tickets',
                 '3 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt="[user] I'm looking for something fun to do. [system] Which city and what type of event? [user] How about a play in London. [system] There are 10 that may interest you. How about A Right Royale Tea at Amba Hotel Charing Cross on March 14th hat 7 pm? [user] Great, can you reserve tickets? [system] Sure, how many? [user] 2 tickets, please, my love.",
            slots=OrderedDict([
                ('city', 'London'), ('date', 'March 14th'),
                ('event_name', 'A Right Royale Tea'),
                ('event_type', 'Theater of possible values Music, Theater'),
                ('number_of_tickets',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ])),
        Prompt(
            utt='[user] Could you find me a musical performance in Sacramento, CA on march 10th? [system] I have 5 options for you. How about the aftershock music festival happening at discovery park at 5 pm on march 10th? [user] Sounds good to me, could you find a hotel in that area? [system] I found 10 hotels in that area. How about Courtyard by Marriott Sacramento Cal Expo, it has 3 stars [user] Sounds good, can I purchase just one ticket to that event?',
            slots=OrderedDict([
                ('city', 'Sacramento'), ('date', 'March 10th'),
                ('event_name', 'aftershock music festival'),
                ('event_type', 'Music of possible values Music, Theater'),
                ('number_of_tickets',
                 '1 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9')
            ]))
    ],
    'Flights_1': [
        Prompt(
            utt='[user] Please search flights for me. I m leaving from Las Vegas on the 6th of this month and coming back on the 9th of this month. [system] Where are you going? [user] I am going to SD. There is only 1 in my group. Make sure to book only refundable tickets. Either business or economy is fine. [system] I found a convenient Southwest Airlines flight for your trip. It leaves at 3:10 pm and the return flight leaves at 8:25 pm. The ticket price is $95. [user] That flight sounds good to me.',
            slots=OrderedDict([
                ('departure_date', '6th of this month'),
                ('origin_city', 'Las Vegas'),
                ('return_date', '9th of this month'),
                ('destination_city', 'SD'),
                ('passengers', '1 of possible values 1, 2, 3, 4'),
                ('airlines',
                 'Southwest Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('inbound_departure_time', '8:25 pm'),
                ('outbound_departure_time', '3:10 pm'),
                ('refundable', 'True of possible values True, False'),
                ('seating_class',
                 'dontcare of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt='[user] Can you find an economy flight ticket for me? [system] When do you plan on coming back? [user] Leaving on March 8th, returning the 13th of March. [system] Where are you departing from? [user] I am departing from SD. I may change my plans, so I want refundable tickets only. [system] Okay, where are you planning to visit? [user] Going to LA. [system] An American Airlines flight is available. It departs at 9:50 am and the return flight leaves at 7:20 am. A ticket costs $68. [user] Sound good to me, please book 4 tickets.',
            slots=OrderedDict([
                ('airlines',
                 'American Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'March 8th'),
                ('destination_city', 'LA'), ('inbound_departure_time',
                                             '7:20 am'), ('origin_city', 'SD'),
                ('outbound_departure_time',
                 '9:50 am'), ('passengers', '4 of possible values 1, 2, 3, 4'),
                ('refundable', 'True of possible values True, False'),
                ('return_date',
                 '13th of March'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt='[user] Can you find me one way flights for my upcoming trip? [system] When do you plan to go? Where will you be leaving from and at what location to you plan to visit? [user] I will be traveling to Seattle and leaving from SD. I will be heading out on 9th of March. I prefer nonrefundable Economy. [system] I have located 1 Delta flight for $189 that could work. The outbound flight takes off at 7:45 am, with 1 stop. Return take off is at 5 pm [user] That sounds like a good flight. Book for a group of 3, returning 11th of March',
            slots=OrderedDict([
                ('airlines',
                 'Delta Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', '9th of March'),
                ('destination_city', 'Seattle'),
                ('inbound_departure_time', '7:45 am'), ('origin_city', 'SD'),
                ('outbound_departure_time',
                 '5 pm'), ('passengers', '3 of possible values 1, 2, 3, 4'),
                ('refundable', 'False of possible values True, False'),
                ('return_date',
                 '11th of March'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt='[user] Find me round economy round trip flights to Portland, OR. [system] Whats the city of departure? [user] Leaving from Los Angeles [system] when are you leaving and coming back? [user] leaving on this saturday and coming back on wednesday next week [system] I have 5 flights. The alaska airlines flight leave at 7:25 am and has return on 1:55 pm and costs $172 [user] that is good [system] Should i book it? [user] yes please book it as refundable tickets for four people.',
            slots=OrderedDict([
                ('airlines',
                 'Alaska Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'this saturday'),
                ('destination_city', 'Portland, OR'),
                ('inbound_departure_time', '1:55 pm'),
                ('origin_city',
                 'Los Angeles'), ('outbound_departure_time', '7:25 am'),
                ('passengers', '4 of possible values 1, 2, 3, 4'),
                ('refundable', 'True of possible values True, False'),
                ('return_date',
                 'wednesday next week'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt='[user] I need to search round trip flights on the 7th. [system] When will you be returning? [user] I will be returning on the 9th of this month. [system] Where will you be coming from and where do want to go? [user] I will leaving from NY going to Seattle, WA. I would like to return on March 12th now. [system] Found 3 flights, one with Delta Airlines that leaves at 6 am. The return flight leaves at 3:15 pm. [user] That sounds great. [system] Would you like to make reservations? [user] Yes 2 tickets, but want only tickets that are refundable as my plans may change.',
            slots=OrderedDict([
                ('airlines',
                 'Delta Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'March 7th'),
                ('destination_city', 'Seattle, WA'),
                ('inbound_departure_time', '3:15 pm'), ('origin_city', 'NY'),
                ('outbound_departure_time',
                 '6 am'), ('passengers', '2 of possible values 1, 2, 3, 4'),
                ('refundable', 'True of possible values True, False'),
                ('return_date',
                 'March 12th'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ]))
    ],
    'Flights_2': [
        Prompt(
            utt='[user] I am looking for a flight from Seattle that return on the 13th. [system] Where are you planning to go? What date would you like to travel? [user] I am flying to SF on the 9th of this month and want one Economy class ticket. [system] I found 3 flights for you. Would you like to fly with Delta Airlines? The flight takes off at 6:10 am and returns 8 am. The ticket costs $131. [user] Ok.',
            slots=OrderedDict([
                ('airlines',
                 'Delta Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date',
                    '9th of this month'), ('destination',
                                           'SF'), ('origin',
                                                   'Seattle'),
                ('passengers',
                 '1 of possible values 1, 2, 3, 4'), ('return_date',
                                                      'the 13th'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt='[user] get me economy tickets from san Fran to Los Angeles on Tuesday next week [system] is the return on March 5th? [user] no, its on 8th of this month [system] ok how about an American airlines which leaves at 3:50 pm and the return is 6:30 am and costs $96 [user] ok, book 1 seat, thanks',
            slots=OrderedDict([
                ('airlines',
                 'American Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'Tuesday next week'),
                ('destination',
                 'Los Angeles'), ('origin', 'san Fran'),
                ('passengers', '1 of possible values 1, 2, 3, 4'),
                ('return_date',
                 '8th of this month'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt='[user] I need help in getting round trip economy flights. [system] What date do you want to travel and the city of departure? [user] yes make it form Lax for Monday next week [system] Returning the day after? [user] No 14th of march, one ticket [system] Ok, i got 5 flights of which an Alaska Airlines flight takes off at 5:15 am and returns 10:10 am and the cost is $131 [user] ok, thanks, thats it',
            slots=OrderedDict([
                ('airlines',
                 'Alaska Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date',
                    'Monday next week'), ('destination', 'Portland'),
                ('origin',
                 'Lax'), ('passengers', '1 of possible values 1, 2, 3, 4'),
                ('return_date',
                 '14th of march'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt='I have a concert in Phoenix and I would need a round flight trip to get there. I want 4 economy tickets from Los Angeles on March 5th. [system] You want to return on March 6th? [user] I want to return next Friday. [system] There are 3 available flights. You want an American Airlines flight? There is one leaving at 7:50 am and arriving at 7:35 am. The cost is $108 [user] Thanks.',
            slots=OrderedDict([
                ('airlines',
                 'American Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'March 5th'), ('destination', 'Phoenix'),
                ('origin', 'Los Angeles'),
                ('passengers', '4 of possible values 1, 2, 3, 4'),
                ('return_date',
                 'next Friday'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ])),
        Prompt(
            utt="[user] Hi, I'm looking for a flight. [system] What city would you like to depart from? [user] I'm flying to Washington and looking to find flights there from LA. I want to fly with American Airlines please. [system] What day would you like to take the return flight? [user] I need Economy class tickets that return on the 12th of March. [system] What date would you like to travel there? [user] I want to go on Sunday this week, and I need three tickets. [system] Found one that departs at 2:10 pm and the return flight takes off at 2:50 pm. [user] That works well.",
            slots=OrderedDict([
                ('airlines',
                 'American Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'Sunday this week'),
                ('destination',
                 'Washington'), ('origin',
                                 'LA'),
                ('passengers', '3 of possible values 1, 2, 3, 4'),
                ('return_date',
                 '12th of March'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                )
            ]))
    ],
    'Flights_3': [
        Prompt(
            utt='[user] I need a round trip flight and I have 0 bags that need to be checked in. [system] On which date would you like to travel and to where? [user] I want to fly to Seattle on the 6th of March and return on the 8th of this month. [system] What city do you want to leave from? [user] Look for four tickets on economy flights departing Atlanta. [system] I have 1 flight that I think you would like. How about Southwest Airlines? This flight leaves at 7:15 pm mand returns at 6:55 pm local time. It costs $485 per person. [user] Oky. Thank you for your assistance.',
            slots=OrderedDict([
                ('airlines',
                 'Southwest Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', '6th of March'),
                ('destination_city', 'Seattle'),
                ('flight_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                ), ('number_checked_bags', '0 of possible values 0, 1, 2'),
                ('origin_city', 'Atlanta'),
                ('passengers', '1 of possible values 1, 2, 3, 4'),
                ('return_date', '8th of this month')
            ])),
        Prompt(
            utt='[user] Can you search for a two way premium economy United flight? [system] What will be your destination? [user] My destination is San Francisco. [system] What will be the city and date of departure? [user] My city and date of departure is New York and on the 1st, retuning on the 3rd. 1 seat and no checked bags [system] I have one which takes off at 11:15 am with 0 layovers and the returning flight is at 8:05 pm [user] Okay, cool',
            slots=OrderedDict([
                ('airlines',
                 'United Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'the 1st'),
                ('destination_city', 'San Francisco'),
                ('flight_class',
                 'Premium Economy of possible values Economy, Premium Economy, Business, First Class'
                ), ('number_checked_bags', '0 of possible values 0, 1, 2'),
                ('origin_city', 'New York'),
                ('passengers', '1 of possible values 1, 2, 3, 4'),
                ('return_date', 'the 3rd')
            ])),
        Prompt(
            utt="[user] Can you tell me the cost of a round trip flight from San Diego to LA on Alaska Airlines leaving on the 13th, 2 Economy tickets. [system] Will you be leaving on March 6th or on March 5th? [user] I will fly out on the 5th of March and return on March 11th with 0 bags. [system] I have 1 Alaska Airlines flight that departs at 9:55 am and the returning flight departs at 6:55 am for $119 per person. [user] That's not bad, thanks!",
            slots=OrderedDict([
                ('airlines',
                 'Alaska Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', '5th of March'),
                ('destination_city', 'LA'),
                ('flight_class',
                 'Economy of possible values Economy, Premium Economy, Business, First Class'
                ), ('number_checked_bags', '0 of possible values 0, 1, 2'),
                ('origin_city', 'San Diego'),
                ('passengers', '2 of possible values 1, 2, 3, 4'),
                ('return_date', 'March 11th')
            ])),
        Prompt(
            utt='[user] Can you find me a round trip flight going NYC from Washington? I need one ticket, have 0 bags and would like a Premium Economy class ticket. I want to leave the 2nd of this month and return the 14th of March. [system] I found 2 flights. There is a flight with American Airlines leaving at 4:10 am local time and returning at 2:10 pm local time for $240 per passenger. How does this sound? [user] That sounds alright.',
            slots=OrderedDict([
                ('airlines',
                 'American Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', '2nd of this month'),
                ('destination_city', 'NYC'),
                ('flight_class',
                 'Premium Economy of possible values Economy, Premium Economy, Business, First Class'
                ), ('number_checked_bags', '0 of possible values 0, 1, 2'),
                ('origin_city', 'Washington'),
                ('passengers', '1 of possible values 1, 2, 3, 4'),
                ('return_date', '14th of March')
            ])),
        Prompt(
            utt='[user] Hi, could you get me round trip premium economy tickets from SD to Los Angeles? No bags to check in. [system] Will you depart on the March 2nd? [user] Yes, and I will return on the 11th of this month. Could you get me Southwest Airlines flight please? [system] Sure, how about 1 Southwest Airlines flight at 11:45 am which returns at 3:35 pm and costs $178 per ticket? [user] Sure, get me 3 tickets.',
            slots=OrderedDict([
                ('airlines',
                 'Southwest Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France'
                ), ('departure_date', 'March 2nd'),
                ('destination_city', 'Los Angeles'),
                ('flight_class',
                 'Premium Economy of possible values Economy, Premium Economy, Business, First Class'
                ), ('number_checked_bags', '0 of possible values 0, 1, 2'),
                ('origin_city', 'SD'),
                ('passengers', '3 of possible values 1, 2, 3, 4'),
                ('return_date', '11th of this month')
            ]))
    ],
    'Flights_4': [
        Prompt(
            utt='[user] I would like to get two tickets on a round trip flight that will get me to San Francisco from San Diego. [system] Will you be leaving on March 2nd? [user] I would like to leave next Monday. [system] What day do you want your return flight to be? [user] I would like to fly with United Airlines and come back on the 11th of this month. I have no preference about which ticket class it is.',
            slots=OrderedDict([
                ('destination_airport', 'San Francisco'),
                ('number_of_tickets', '2 of possible values 1, 2, 3, 4'),
                ('origin_airport', 'San Diego'),
                ('departure_date', 'next Monday'),
                ('airlines',
                 'United Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France, South African Airways, LOT Polish Airlines, LATAM Brasil'
                ), ('return_date', '11th of this month'),
                ('seating_class',
                 'dontcare of possible values Economy, Premium Economy, Business'
                )
            ])),
        Prompt(
            utt='[user] Great I need 3 round trip tickets from Chicago to Los Angeles, any class or airline. [system] Which day would you like the return flight? [user] I am returning on the 12th of this month. [system] When are you leaving? [user] I want to travel on the 5th.',
            slots=OrderedDict([
                ('airlines',
                 'dontcare of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France, South African Airways, LOT Polish Airlines, LATAM Brasil'
                ), ('departure_date', 'the 5th'),
                ('destination_airport', 'Los Angeles'),
                ('number_of_tickets', '3 of possible values 1, 2, 3, 4'),
                ('origin_airport',
                 'Chicago'), ('return_date',
                              '12th of this month'),
                ('seating_class',
                 'dontcare of possible values Economy, Premium Economy, Business'
                )
            ])),
        Prompt(
            utt="[user] I'll be going to see Alma Deutscher in New York, could you help me find two round trip economy class tickets with United Airlines heading there from Philly? [system] And which date are you going to return? [user] I will be travelling from Sunday this week to March 6th. [system] Would you like a direct United Airlines flight departing at 11:30 am and returning at 5:35 am costing $99? [user] Sounds great!",
            slots=OrderedDict([
                ('airlines',
                 'United Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France, South African Airways, LOT Polish Airlines, LATAM Brasil'
                ), ('departure_date', 'Sunday this week'),
                ('destination_airport', 'New York'),
                ('number_of_tickets', '2 of possible values 1, 2, 3, 4'),
                ('origin_airport',
                 'Philly'), ('return_date',
                             'March 6th'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business'
                )
            ])),
        Prompt(
            utt="[user] Can you check out economy class tickets that are going to NY? I'll need 4 tickets and would prefer flying with United Airlines from Washington. [system] when are you going and returning? [user] I will be going the day after tomorrow and come back on the 14th of March. [system] There is 1 flight. Is United Airlines good? There is a direct onward flight that leaves at 4:35 pm and the return flight will leave at 10:15 am. [user] That works.",
            slots=OrderedDict([
                ('airlines',
                 'United Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France, South African Airways, LOT Polish Airlines, LATAM Brasil'
                ), ('departure_date',
                    'day after tomorrow'), ('destination_airport', 'NY'),
                ('number_of_tickets', '4 of possible values 1, 2, 3, 4'),
                ('origin_airport',
                 'Washington'), ('return_date',
                                 '14th of March'),
                ('seating_class',
                 'Economy of possible values Economy, Premium Economy, Business'
                )
            ])),
        Prompt(
            utt="[user] please find me a round-trip flight to Portland. It should be on United Airlines, but the ticket type isn't important. [system] What's your preferred travel and return date? [user] I'd like to leave today and return on 14th of this month. [system] There's a direct outbound United Airlines flight leaving at 11:40 pm. The inbound flight takes off at 4:40 am, and the cost per ticket is $104. How does that sound? [user] That works for me. Please find me 4 tickets there and also hotels in that area.",
            slots=OrderedDict([
                ('airlines',
                 'United Airlines of possible values United Airlines, American Airlines, Delta Airlines, Southwest Airlines, Alaska Airlines, British Airways, Air Canada, Air France, South African Airways, LOT Polish Airlines, LATAM Brasil'
                ), ('departure_date', 'today'),
                ('destination_airport', 'Portland'),
                ('number_of_tickets', '4 of possible values 1, 2, 3, 4'),
                ('origin_airport',
                 'Seattle, WA'), ('return_date', '14th of this month'),
                ('seating_class',
                 'dontcare of possible values Economy, Premium Economy, Business'
                )
            ]))
    ],
    'Homes_1': [
        Prompt(
            utt='[user] I am looking for a funished apartment with one bath. [system] Which area and how many bedrooms do you want in the apartment? [user] I want this apartment in Santa Rosa, with three bedrooms. Is there any place where pets are allowed? [system] There are 3 apartments in Marlow Apartments at 3076 Marlow Road, with 3 bedrooms, 1 bath and the rent is 4300 dollars per month. [user] Yes, I like it. I want to schedule a visit on the 14th.',
            slots=OrderedDict([
                ('furnished', 'True of possible values True, False'),
                ('number_of_baths', '1 of possible values 1, 2, 3, 4'),
                ('area', 'Santa Rosa'),
                ('number_of_beds', '3 of possible values 1, 2, 3, 4'),
                ('pets_allowed', 'True of possible values True, False'),
                ('property_name', 'Marlow Apartments'),
                ('visit_date', 'March 14th')
            ])),
        Prompt(
            utt='[user] I need to find a furnished apartment in Antioch with 1 bath where pets are allowed. [system] How many bedrooms do you need? [user] I need 1 bedroom. [system] There is one located at 5100 Vista Grande Dr, called Cross Pointe with 1 bath and 1 bedroom. It cost $1,500 a month. [user] That sounds great. I would like to schedule an appointment to view the apartment on the 12th of March.',
            slots=OrderedDict([
                ('area', 'Antioch'),
                ('furnished', 'True of possible values True, False'),
                ('number_of_baths', '1 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '1 of possible values 1, 2, 3, 4'),
                ('pets_allowed', 'True of possible values True, False'),
                ('property_name', 'Cross Pointe'),
                ('visit_date', '12th of March')
            ])),
        Prompt(
            utt='[user] I need help in finding a 4 bed three bath furnished apartment in Richmond. Something pet-friendly [system] I found 1 apartment matching your specifications, baycliff apartments. It is located at 2300 lancaster drive, available to rent for $4850 per month. [user] Okay, sounds good [system] Should I schedule a property visit? [user] yes, can we do March 9th?',
            slots=OrderedDict([
                ('area', 'Richmond'),
                ('furnished', 'True of possible values True, False'),
                ('number_of_baths', '3 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '4 of possible values 1, 2, 3, 4'),
                ('pets_allowed', 'True of possible values True, False'),
                ('property_name', 'baycliff apartments'),
                ('visit_date', 'March 9th')
            ])),
        Prompt(
            utt="[user] Can you find me a 2 bed 3 bath apartment in Sunnyvale? I can't care less about pets [system] I have found 2 apartments with 2 bedrooms and 3 bathrooms at Renaissance Apartment Homes located at 718 Old San Francisco Road. The rent is 3950 dollars per month. [user] Could I have their telephone number? [system] Their phone number is 408-736-1600. [user] That sounds like what I am looking for. [user] I would like to schedule a viewing on the 7th of March if we could do that.",
            slots=OrderedDict([
                ('area', 'Sunnyvale'),
                ('furnished', 'True of possible values True, False'),
                ('number_of_baths', '3 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '2 of possible values 1, 2, 3, 4'),
                ('pets_allowed', 'dontcare of possible values True, False'),
                ('property_name', 'Renaissance Apartment Homes'),
                ('visit_date', '7th of March')
            ])),
        Prompt(
            utt="[user] Hi, I'd like help finding an apartment. [system] Sure, how many bedrooms do you need? [user] I just want a one bedroom place. [system] Okay, what area are you looking in? [user] I'd like a furnished apartment in Hayward. I have a dog and just need one bath. [system] You could look into Cypress House Apartments at 24955 Cypress Avenue. This is going for $1,850 a month. [user] Yeah, that sounds great. I want to see the place on March 7th.",
            slots=OrderedDict([
                ('area', 'Hayward'),
                ('furnished', 'True of possible values True, False'),
                ('number_of_baths', '1 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '1 of possible values 1, 2, 3, 4'),
                ('pets_allowed', 'True of possible values True, False'),
                ('property_name', 'Cypress House Apartments'),
                ('visit_date', 'March 7th')
            ]))
    ],
    'Homes_2': [
        Prompt(
            utt='[user] I want to find a place to rent with an in-unit laundry and a garage. [system] How many bedrooms and baths and which location? [user] I would like four bedrooms and 3 bathrooms in Walnut Creek. Actually, I would prefer to buy instead. [system] I found 1 at Diablo Pointe Apartments at 1450 Creekside Drive with a price of $4,350,000. [user] I would like to visit the property on the 11th of this month.',
            slots=OrderedDict([
                ('has_garage', 'True of possible values True, False'),
                ('in_unit_laundry', 'True of possible values True, False'),
                ('intent', 'buy of possible values rent, buy'),
                ('area', 'Walnut Creek'),
                ('number_of_baths', '3 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '4 of possible values 1, 2, 3, 4'),
                ('property_name', 'Diablo Pointe Apartments'),
                ('visit_date', '11th of this month')
            ])),
        Prompt(
            utt="[user] Will you find a place buy with a garage and 2 baths? [system] How many bedrooms? Which area? [user] I prefer 2 bed rooms in Vacaville with in-unit laundry. [system] Found 2 houses. There's Meadowood Apartments located at 199 Aegean Way for $2,700,000. [user] Great. [system] Schedule a visit? [user] I would like to visit on the 2nd of this month.",
            slots=OrderedDict([
                ('area', 'Vacaville'),
                ('has_garage', 'True of possible values True, False'),
                ('in_unit_laundry', 'True of possible values True, False'),
                ('intent', 'buy of possible values rent, buy'),
                ('number_of_baths', '2 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '2 of possible values 1, 2, 3, 4'),
                ('property_name', 'Meadowood Apartments'),
                ('visit_date', '2nd of this month')
            ])),
        Prompt(
            utt="[user] I'm looking to move into a new place to live in Antioch. I'm flexible on the property itself, but it must have an in-unit laundry, a garage, 2 each of bedrooms and bathrooms. [system] Will you be renting or buying the property? [user] I think I'll go with buy. [system] There are 3 very nice homes available. The Magnolia Apartments located at 1015 Fitzuren Road #7 has a price tag of $2,550,000. [user] A bit pricey but that should be within my budget. Sounds good to me. Check for a visit on the 2nd of this month.",
            slots=OrderedDict([
                ('area', 'Antioch'),
                ('has_garage', 'True of possible values True, False'),
                ('in_unit_laundry', 'True of possible values True, False'),
                ('intent', 'buy of possible values rent, buy'),
                ('number_of_baths', '2 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '2 of possible values 1, 2, 3, 4'),
                ('property_name', 'Magnolia Apartments'),
                ('visit_date', '2nd of this month')
            ])),
        Prompt(
            utt='[user] I need a place to rent with a garage and two bathrooms. [system] Where should I look and what number of bed rooms will work? [user] I need at least two bedrooms and you can look in Walnut Creek, inside private laudry is mandatory. [system] Montecito Pines Apartments would work for $2,700. [user] I like that one. [system] Should I make an appointment for you to view the property? [user] Absolutely,for the 9th.',
            slots=OrderedDict([
                ('area', 'Walnut Creek'),
                ('has_garage', 'True of possible values True, False'),
                ('in_unit_laundry', 'True of possible values True, False'),
                ('intent', 'rent of possible values rent, buy'),
                ('number_of_baths', '2 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '2 of possible values 1, 2, 3, 4'),
                ('property_name', 'Montecito Pines Apartments'),
                ('visit_date', 'the 9th')
            ])),
        Prompt(
            utt="[user] My lease is running out, I need another place to rent in Mountain View. [system] How many beds and baths? [user] One of each, I will be by myself [system] Thoughts on in-unit washer dryer or a garage? [user] I need neither, they're expensive. [system] With your budget I could only find The Arbors who have something for $1400. [user] Thanks! That'll do. When can I visit them? Maybe tomorrow?",
            slots=OrderedDict([
                ('has_garage', 'False of possible values True, False'),
                ('in_unit_laundry', 'False of possible values True, False'),
                ('intent', 'rent of possible values rent, buy'),
                ('area', 'Mountain View'),
                ('number_of_baths', '1 of possible values 1, 2, 3, 4'),
                ('number_of_beds', '1 of possible values 1, 2, 3, 4'),
                ('property_name', 'The Arbors'), ('visit_date', 'tomorrow')
            ]))
    ],
    'Hotels_1': [
        Prompt(
            utt="[user] I need one 1-star hotel room with WiFi please. [system] What city would you like to stay? [user] I need it in San Francisco [system] I have 8 options for you. Beck's Motor Lodge is a 1 star. [user] That sounds good. Can you reserve for 2 days? [system] When would you like to check in? [user] I need it on the 11th of March.",
            slots=OrderedDict([
                ('has_wifi', 'True of possible values True, False'),
                ('destination', 'San Francisco'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('star_rating', '1 of possible values 1, 2, 3, 4, 5'),
                ('hotel_name', "Beck's Motor Lodge"), ('number_of_days', '2'),
                ('check_in_date', '11th of March')
            ])),
        Prompt(
            utt='[user] Will you find a three star hotel in Sacramento with wifi? [system] I got here Courtyard By Marriott Sacramento Midtown, a 3 star hotel. [user] Great. [system] Want to reserve a room? [user] Yes. 1 room for 1 day with check in on the 3rd.',
            slots=OrderedDict([
                ('check_in_date', 'March 3rd'), ('destination', 'Sacramento'),
                ('has_wifi', 'True of possible values True, False'),
                ('hotel_name', 'Courtyard By Marriott Sacramento Midtown'),
                ('number_of_days', '1'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('star_rating', '3 of possible values 1, 2, 3, 4, 5')
            ])),
        Prompt(
            utt='[user] 2 star hotel please in NY, internet or not [system] 1 hotel brooklyn bridge is a top choice for 2-star hotels there. Dates? [user] I will arrive on March 8th and will stay for 2 days [system] 1 hotel brooklyn bridge March 8th for 2 days in NY for 1 room, that good? [user] yes that is fine',
            slots=OrderedDict([
                ('check_in_date', 'March 8th'), ('destination', 'NY'),
                ('has_wifi', 'dontcare of possible values True, False'),
                ('hotel_name', '1 hotel brooklyn bridge'),
                ('number_of_days', '2'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('star_rating', '2 of possible values 1, 2, 3, 4, 5')
            ])),
        Prompt(
            utt='[user] I need a hotel room with wifi in Phoenix [system] I found 10, Cambria Hotel North Scottsdale Desert Ridge is a nice 2 star. [user] That is great, I need a reservation for 1 room. [system] How many days do you need it for? [user] Check in March 9th for 4 days.',
            slots=OrderedDict([
                ('check_in_date', 'March 9th'), ('destination', 'Phoenix'),
                ('has_wifi', 'True of possible values True, False'),
                ('hotel_name', 'Cambria Hotel North Scottsdale Desert Ridge'),
                ('number_of_days', '4'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('star_rating', '2 of possible values 1, 2, 3, 4, 5')
            ])),
        Prompt(
            utt='[user] I would like to see the Lakers game in Los Angeles and need a hotel there. [system] Starting when? [user] March 1st. I would like to book 2 rooms in a three star hotel that has wifi. [system] Airtel Plaza hotel is a nice 3 star hotel close to the venue. [user] Yes that works I would like to make a reservation for 1 day.',
            slots=OrderedDict([
                ('check_in_date', 'March 1st'), ('destination', 'Los Angeles'),
                ('has_wifi', 'True of possible values True, False'),
                ('hotel_name', 'Airtel Plaza Hotel'), ('number_of_days', '1'),
                ('number_of_rooms', '2 of possible values 1, 2, 3'),
                ('star_rating', '3 of possible values 1, 2, 3, 4, 5')
            ]))
    ],
    'Hotels_2': [
        Prompt(
            utt="[user] I'm looking for a house to stay in for the night. [system] Sure! Where do you plan to visit? [user] I'll be going to New York. [system] Ok, there's a house open on 100 Greenwich Street, the rating is 4.4. [user] Actually, we'd like to go to San Francisco instead, need something for 5 people with laundry service and it would be great if the rating is 4.3 or better! [system] Ok sure! 9 houses are available, there is one at 1100 Market Street Hotel entrance is at, 45 McAllister Street with a rating of 4.3 [user] That looks perfect. I'd like to book from the 2nd of March through March 14th if possible.",
            slots=OrderedDict([
                ('where_to', 'San Francisco'),
                ('has_laundry_service', 'True of possible values True, False'),
                ('number_of_adults', '5 of possible values 1, 2, 3, 4, 5'),
                ('rating', '4.3'), ('check_out_date', 'March 14th'),
                ('check_in_date', '2nd of March')
            ])),
        Prompt(
            utt='[user] Will you check the weather on March 7th in Portland? [system] Average temperature is 72 degrees with 32 percent chance of rain. [user] Great. Will you find a house for 2 with laundry service and 4.5 or higher rating? [system] 1150 Northwest 9th Avenue with rating of 4.5. [user] Great. Wanna check in March 7th and out March 10th.',
            slots=OrderedDict([
                ('check_in_date', 'March 7th'),
                ('check_out_date', 'March 10th'),
                ('has_laundry_service', 'True of possible values True, False'),
                ('number_of_adults', '2 of possible values 1, 2, 3, 4, 5'),
                ('rating', '4.5'), ('where_to', 'Portland')
            ])),
        Prompt(
            utt='[user] Find for me a house to stay with minimum rating 3.9.I want to be for 1 people and with laundry service available. [system] On which city you plan to stay? [user] I travel to SD. [system] With rating 3.9 is a nice house located at 1003 Coast Boulevard, La Jolla, California 92037, United States. [user] Sounds very good, reserve from 3rd of this month and check out March 4th for just 1.',
            slots=OrderedDict([
                ('check_in_date', '3rd of this month'),
                ('check_out_date', 'March 4th'),
                ('has_laundry_service', 'True of possible values True, False'),
                ('number_of_adults', '1 of possible values 1, 2, 3, 4, 5'),
                ('rating', '3.9'), ('where_to', 'SD')
            ])),
        Prompt(
            utt='[user] I want to find a house to stay for 1 in London [system] There is a lovely house with a 4.3 rating on 1 Addington Street. [user] What else is there with at least 3.7 rating and which has a laundry service. [system] There is one with a 3.7 rating at 1 Baffin Way. [user] Gotcha, I would like to book it. [system] From when to when? [user] Checking in tomorrow and departing on 3rd of March.',
            slots=OrderedDict([
                ('check_in_date', 'tomorrow'),
                ('check_out_date', '3rd of March'),
                ('has_laundry_service', 'True of possible values True, False'),
                ('number_of_adults', '1 of possible values 1, 2, 3, 4, 5'),
                ('rating', '3.7'), ('where_to', 'London')
            ])),
        Prompt(
            utt='[user] I want to find a house to stay in NYC for 1 person with laundry service and the rating should be 3.7 or more. [system] There is a house at 100 Greenwich Street with a rating of 3.7. [user] That sounds good. Can I arrive there the 7th of this month. [system] When will you be checking out? [user] I will leave on the 13th.',
            slots=OrderedDict([
                ('check_in_date', '7th of this month'),
                ('check_out_date', 'the 13th'),
                ('has_laundry_service', 'True of possible values True, False'),
                ('number_of_adults', '1 of possible values 1, 2, 3, 4, 5'),
                ('rating', '3.7'), ('where_to', 'NYC')
            ]))
    ],
    'Hotels_3': [
        Prompt(
            utt="[user] Can you look for hotels for me? [system] Where would you like to stay? [user] I'm looking for hotels in Vancouver. I need to be able to have pets stay with me. [system] Alright, what about Best Western Plus Chateau Granville Hotel & Suites & Conference Ctr. ? It has a 4.1 rating. [user] That's great. [system] Would you like me to go ahead and reserve you a room? [user] Yes please, make the reservation, checking in 10th of March and out March 13th. [system] Alright, and how many rooms? [user] I'd need 1.",
            slots=OrderedDict([
                ('location', 'Vancouver'),
                ('pets_welcome', 'True of possible values True, False'),
                ('hotel_name',
                 'Best Western Plus Chateau Granville Hotel & Suites & Conference Ctr.'
                ), ('check_in_date', '10th of March'),
                ('check_out_date', 'March 13th'),
                ('number_of_rooms', '1 of possible values 1, 2, 3')
            ])),
        Prompt(
            utt="[user] Find me a hotel in LA which allows pets [system] How about Airtel Plaza Hotel which has 3.9 rating [user] That's a good one [system] Can I reserve the hotel? [user] Yes, a room in that hotel from March 6th through the 12th",
            slots=OrderedDict([
                ('check_in_date', 'March 6th'), ('check_out_date', 'the 12th'),
                ('hotel_name', 'Airtel Plaza Hotel'), ('location', 'LA'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('pets_welcome', 'True of possible values True, False')
            ])),
        Prompt(
            utt="[user] I'm looking for hotels in London, England [system] I suggest 45 Park Lane. It has a rating of 4.7 [user] I'm allergic to pets. Do you have a hotel where pets are disallowed? [system] Yes, I have Ace Hotel with a rating of 4.3 [user] Thats perfect. I want to reserve two rooms. Check in on 6th of March and check out on 11th of March",
            slots=OrderedDict([
                ('check_in_date', '6th of March'),
                ('check_out_date', '11th of March'),
                ('hotel_name', 'Ace Hotel'), ('location', 'London, England'),
                ('number_of_rooms', '2 of possible values 1, 2, 3'),
                ('pets_welcome', 'False of possible values True, False')
            ])),
        Prompt(
            utt="[user] Can you find me a hotel in Vancoucer? Starting 2nd of march until march 4th. [system] I've heard good things about atrium inn vancouver rated 3.7 [user] are they cat friendly? [system] yes they are cool with that. wanna reserve it? [user] I want one room, yes",
            slots=OrderedDict([
                ('check_in_date', '2nd of march'),
                ('check_out_date', 'march 4th'),
                ('hotel_name', 'atrium inn vancouver'),
                ('location', 'Vancouver'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('pets_welcome', 'True of possible values True, False')
            ])),
        Prompt(
            utt="[user] Could you search me hotels around London, England where pets are allowed and i wish to book one room. [system] What is your opinion about 45 park lane with 4.7 rating? [user] Okay this is perfect. I'd wanna book. [system] I know from your flights you're arriving on march 7th and leaving 11th of march. Are those dates good for the hotel too? [user] Yes, you got it.",
            slots=OrderedDict([
                ('check_in_date', 'march 7th'),
                ('check_out_date', '11th of march'),
                ('hotel_name', '45 park lane'), ('location', 'London'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('pets_welcome', 'True of possible values True, False')
            ]))
    ],
    'Hotels_4': [
        Prompt(
            utt="[user] I need to find a hotel, but it's got to be a 3 star hotel. Nothing else will work. [system] Okay, I can find you some 3 star hotels, but I'll need to know in which city you would like me to search. [user] In London, please. I'd love to be able to smoke in the apartment, but I understand not many hotels are offering that nowadays. [system] There is this hotel called Assembly Hotel London, which is also a 3 star hotel. How does this sound? [user] It sounds promising, I'd like to make a reservation. [system] When would you like the check in date to be? [user] Make the check in date Tuesday next week. [system] Okay, and how long would you like to be staying for? [user] Make the reservation for 8 days, and two rooms, for my plans are always changing.",
            slots=OrderedDict([
                ('star_rating', '3 of possible values 1, 2, 3, 4, 5'),
                ('location', 'London'),
                ('smoking_allowed', 'True of possible values True, False'),
                ('place_name', 'Assembly Hotel London'),
                ('check_in_date', 'Tuesday next week'), ('stay_length', '8'),
                ('number_of_rooms', '2 of possible values 1, 2, 3')
            ])),
        Prompt(
            utt="[user] I need two rooms in a hotel in new york. I'll be reaching friday next week. Know of any good ones? [system] amsterdam court hotel might work for you if you're looking for something close to broadway, it's 3 stars [user] I'll trust you on this then, book 2 rooms for a day please? Does it have smoking rooms? [system] it does actually [user] lovely",
            slots=OrderedDict([
                ('check_in_date', 'friday next week'), ('location', 'new york'),
                ('number_of_rooms', '2 of possible values 1, 2, 3'),
                ('place_name', 'amsterdam court hotel'),
                ('smoking_allowed', 'True of possible values True, False'),
                ('star_rating', '3 of possible values 1, 2, 3, 4, 5'),
                ('stay_length', '1')
            ])),
        Prompt(
            utt="[user] Hello. Can you help me find a three star hotel in London that allows smoking. [system] I've found 10 hotels that match your criteria. How about the 3 star hotel called Abbey Court Hotel (Hyde Park)? [user] Okay, that sounds good. Go ahead and start making the reservation, I will be staying in one room for two days starting Thursday next week.",
            slots=OrderedDict([
                ('check_in_date', 'Thursday next week'), ('location', 'London'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('place_name', 'Abbey Court Hotel (Hyde Park)'),
                ('smoking_allowed', 'True of possible values True, False'),
                ('star_rating', '3 of possible values 1, 2, 3, 4, 5'),
                ('stay_length', '2')
            ])),
        Prompt(
            utt="[user] I want to find one room at a 4 star smoker-friendly hotel in KL. [system] There's a few. Capri By Fraser Kuala Lumpur is a 4 star hotel and quite nice. [user] That sounds good. I want to make the hotel reservation now for 6 days. I'm checking starting this Sunday. How much per night? [system] Can do. The hotel is $160 per night. [user] Thanks so much.",
            slots=OrderedDict([
                ('check_in_date', 'this Sunday'), ('location', 'KL'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('place_name', 'Capri By Fraser Kuala Lumpur'),
                ('smoking_allowed', 'True of possible values True, False'),
                ('star_rating', '4 of possible values 1, 2, 3, 4, 5'),
                ('stay_length', '6')
            ])),
        Prompt(
            utt='[user] Hi i need a help, a relative is coming here to meet me, i am looking to find a 1-star hotel without smoking restrictions [system] Sure, I will help you, in NYC right? [user] Right, one room from Friday next week [system] Central park West Hostel is a pretty good one. How long is your relative staying? [user] for 5 days. Can you book this?',
            slots=OrderedDict([
                ('check_in_date', 'Friday next week'), ('location', 'NYC'),
                ('number_of_rooms', '1 of possible values 1, 2, 3'),
                ('place_name', 'Central park West Hostel'),
                ('smoking_allowed', 'True of possible values True, False'),
                ('star_rating', '1 of possible values 1, 2, 3, 4, 5'),
                ('stay_length', '5')
            ]))
    ],
    'Media_1': [
        Prompt(
            utt='[user] I want to find some movies online. [system] What kind of movies do you like? [user] I like Comic movies best. Laura Steinel has directed some good movies of that genre. [system] I have found 1 movie you may like. It is called Family, would you like to see that one? [user] Family is a good choice, play that with subtitles please.',
            slots=OrderedDict([('directed_by', 'Laura Steinel'),
                               ('genre', 'Comic'), ('title', 'Family'),
                               ('subtitles',
                                'True of possible values True, False')])),
        Prompt(
            utt='[user] I want to find some drama movies online that are directed by Nia DaCosta. [system] Do you like Little Woods? [user] Yes, I want to watch Little Woods right now, that would be great! [system] Please confirm that Little Woods plays without subtitles. [user] Sure!',
            slots=OrderedDict([
                ('directed_by', 'Nia DaCosta'), ('genre', 'drama'),
                ('subtitles', 'False of possible values True, False'),
                ('title', 'Little Woods')
            ])),
        Prompt(
            utt='[user] I want to find online Detective movies directed by Orson Welles. [system] You want Citizen Kane? [user] Citizen Kane is great, i want to watch now, no closed captions.',
            slots=OrderedDict([('directed_by', 'Orson Welles'),
                               ('genre', 'Detective'),
                               ('subtitles',
                                'False of possible values True, False'),
                               ('title', 'Citizen Kane')])),
        Prompt(
            utt='[user] I would like to see a Funny movie online, a movie directed by Morton DaCosta [system] Would you like Auntie Mame? [user] Auntie Mame is ok! [system] What else can I do for you? [user] I would like to watch the movie immediately, with subtitles.',
            slots=OrderedDict([('directed_by', 'Morton DaCosta'),
                               ('genre', 'Funny'),
                               ('subtitles',
                                'True of possible values True, False'),
                               ('title', 'Auntie Mame')])),
        Prompt(
            utt="[user] I'm into Kenyan cinema these days. Anything from there? [system] What genre? [user] I'd like drama. likarion wainaina is apparently a famous director there. [system] I have supa modo available to rent [user] supa modo is good. It's in Swahili, do you need subs? [user] Yes please",
            slots=OrderedDict([('directed_by', 'likarion wainaina'),
                               ('genre', 'drama'),
                               ('subtitles',
                                'True of possible values True, False'),
                               ('title', 'supa modo')]))
    ],
    'Media_2': [
        Prompt(
            utt='[user] Will you find me a movie? [system] What genre? [user] I am looking for a drama directed by Alfred Hitchcock with Alix Talton in it. [system] I found The Man Who Knew Too Much. [user] I would like to rent Man Who Knew Too Much. No subs.',
            slots=OrderedDict([
                ('actors', 'Alix Talton'), ('director', 'Alfred Hitchcock'),
                ('genre', 'drama'), ('movie_name', 'Man Who Knew Too Much'),
                ('subtitle_language',
                 'None of possible values None, English, Mandarin, Spanish')
            ])),
        Prompt(
            utt='[user] Are there any movies directed by Iain Softley and starring Don Hewitt.? [system] You prefer action, history, detective, which one? [user] I Prefer Drama. [system] I think Hackers is 1 you can watch of such genre. [user] That sounds good rent Hackers please. [system] Ok confirm please, Hackers with no subtitles. [user] No , I need it with Spanish subtitles.',
            slots=OrderedDict([
                ('actors', 'Don Hewitt'), ('director', 'Iain Softley'),
                ('genre', 'Drama'), ('movie_name', 'Hackers'),
                ('subtitle_language',
                 'Spanish of possible values None, English, Mandarin, Spanish')
            ])),
        Prompt(
            utt="[user] I want to find a drama to watch starring Stephen Boxer and directed by Trevor Nunn. [system] How about Red Joan? [user] Okay let's rent Red Joan. [system] You want to rent Red Joan with no subtitles, correct? [user] Yes that is correct",
            slots=OrderedDict([
                ('actors', 'Stephen Boxer'), ('director', 'Trevor Nunn'),
                ('genre', 'drama'), ('movie_name', 'Red Joan'),
                ('subtitle_language',
                 'None of possible values None, English, Mandarin, Spanish')
            ])),
        Prompt(
            utt='[user] My mind is getting out,I think its time for watch a movie starred by Larry Jenkins and directed by Brian De Palma. [system] What type of movie are you looking for? [user] I always prefer Detective movies, try for that. [system] There is 1 move for your satisfaction, Body Double. Will it work? [user] Ok go ahead with Body Double, skip the subs',
            slots=OrderedDict([
                ('actors', 'Larry Jenkins'), ('director', 'Brian De Palma'),
                ('genre', 'Detective'), ('movie_name', 'Body Double'),
                ('subtitle_language',
                 'None of possible values None, English, Mandarin, Spanish')
            ])),
        Prompt(
            utt='[user] Find me a Comic movie to see. [system] Dr. Strangelove is a personal favorite of mine. [user] What else do you have? Can you find a movie directed by Harry Elfont starring Mark Seliger? [system] Do you like Josie and the Pussycats? [user] Josie and the Pussycats works for me. [system] Shall I rent the movie now? [user] Yes, rent the movie with no subtitles.',
            slots=OrderedDict([
                ('actors', 'Mark Seliger'), ('director', 'Harry Elfont'),
                ('genre', 'Comic'), ('movie_name', 'Josie and the Pussycats'),
                ('subtitle_language',
                 'None of possible values None, English, Mandarin, Spanish')
            ]))
    ],
    'Media_3': [
        Prompt(
            utt='[user] Could you help me find for Ghost movies featuring Ian Hendry? [system] Would you like to watch Theatre of Blood then? [user] Sure, theatre of blood sounds great! Could you play it for me now? [system] Please confirm your request to play the Theatre of Blood with English subtitles. [user] That is fine with me.',
            slots=OrderedDict([
                ('genre', 'Ghost'), ('starring', 'Ian Hendry'),
                ('title', 'Theatre of Blood'),
                ('subtitle_language',
                 'English of possible values English, Spanish, Hindi, French')
            ])),
        Prompt(
            utt="[user] I'm looking for a Drama movie with Margaret Early that's available online. [system] There's 1 movie that matches. How about Stage Door? [user] I really like Stage Door. Please put that on. [system] I'll play Stage Door with English subtitles, is that correct? [user] Yeahexperience.",
            slots=OrderedDict([
                ('genre', 'Drama'), ('starring', 'Margaret Early'),
                ('subtitle_language',
                 'English of possible values English, Spanish, Hindi, French'),
                ('title', 'Stage Door')
            ])),
        Prompt(
            utt="[user] I need help finding something good to watch online. I feel like family movies featuring Ronald Young. [system] I've found 1 movie you might like. How about Steel Magnolias? [user] I've heard of that. Steel Magnolias sounds great, put that on with English subtitles.",
            slots=OrderedDict([
                ('genre', 'family'), ('starring', 'Ronald Young'),
                ('subtitle_language',
                 'English of possible values English, Spanish, Hindi, French'),
                ('title', 'Steel Magnolias')
            ])),
        Prompt(
            utt='[user] I would like to see a fantasy movie starring Timothy Bateson. [system] There is 1 movie that is suitable called Labyrinth. [user] I want to see Labyrinth right now with French subtitles.',
            slots=OrderedDict([
                ('genre', 'fantasy'), ('starring', 'Timothy Bateson'),
                ('subtitle_language',
                 'French of possible values English, Spanish, Hindi, French'),
                ('title', 'Labyrinth')
            ])),
        Prompt(
            utt='[user] I am curious about Fight movies starring Bronagh Gallagher. [system] Lol you mean Pulp Fiction. [user] That Pulp Fiction is supposed to be good lets see it. [system] To confirm, Play Pulp Fiction with English subtitles. [user] Sure, do it.',
            slots=OrderedDict([
                ('genre', 'Fight'), ('starring', 'Bronagh Gallagher'),
                ('subtitle_language',
                 'English of possible values English, Spanish, Hindi, French'),
                ('title', 'Pulp Fiction')
            ]))
    ],
    'Messaging_1': [
        Prompt(
            utt='[user] Howdy, can you share a location with a friend? [system] Who would you like to share the location with? [user] I want it shared with Kelly. [system] And what address is that? [user] That would be 3810 Mowry Avenue. With Olivia, sorry!',
            slots=OrderedDict([('contact_name', 'Olivia'),
                               ('location', '3810 Mowry Avenue')])),
        Prompt(
            utt='[user] Can you share the address of Great Clips with Jeremy? He has an appointment there [system] Which location of theirs? [user] 20686 Homestead Road',
            slots=OrderedDict([('contact_name', 'Jeremy'),
                               ('location', '20686 Homestead Road')])),
        Prompt(
            utt="[user] I'd like to find a salon. [system] What city should I check? [user] Find a salon in Mill Valley. [system] Ap Luxe Salon is a nice Mill Valley salon. [user] Sounds good. Would you share the location, 40 Millwood Street, with Abigail?",
            slots=OrderedDict([('contact_name', 'Abigail'),
                               ('location', '40 Millwood Street')])),
        Prompt(
            utt="[user] Hey do you know Aya Salon & spa's address? [system] 1880 solano avenue. Need to share that with someone? [user] Yes, send that to Emma please",
            slots=OrderedDict([('contact_name', 'Emma'),
                               ('location', '1880 Solano Avenue')])),
        Prompt(
            utt="[user] Hey can you share my current location with Jack? [system] Maps tells me you're at 1012 Oak Grove Avenue, is that right?[user] Yeah do that.",
            slots=OrderedDict([('contact_name', 'Jack'),
                               ('location', '1012 Oak Grove Avenue')]))
    ],
    'Movies_1': [
        Prompt(
            utt="[user] I want to spend my free time by watching a nice Movie like the one with Dramatic Story. Can you find something for me at Century 20 Oakridge and XD. [system] Any preferred location? [user] Search it in San Jose. [system] Is that Brittany Runs a Marathon okay for you? Or you want to watch Overcomer? [user] I prefer to go with Brittany Runs a Marathon. Please book 2 tickets on March 5th for a 3D show [system] They don't have 3D shows, but they do have a regular show at 10:30 pm that day. How does that sound? [user] That is okay to proceed.",
            slots=OrderedDict([
                ('genre', 'Dramatic Story'),
                ('theater_name', 'Century 20 Oakridge and XD'),
                ('location', 'San Jose'),
                ('movie_name', 'Brittany Runs a Marathon'),
                ('number_of_tickets',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('show_date', 'March 5th'),
                ('show_type', 'regular of possible values regular, 3d, imax'),
                ('show_time', '10:30 pm')
            ])),
        Prompt(
            utt="[user] I want to watch a movie to refresh myself. Can you help me to find a nice 3D movie to watch in San Jose? [system] How about any of 47 Meters down: Uncaged, Ad Astra and After The Wedding. [user] IDK, something supernatural? Oh and at AMC Saratoga. [system] What about your opinion on IT Chapter Two? [user] Yes, IT two has a great trailer. Book me 4 tickets for the 1st of March? [system] They have a 7 pm and a 9:30 pm on that day. [user] I'll take the latter. 3D show, to confirm.",
            slots=OrderedDict([
                ('genre', 'Supernatural'), ('location', 'San Jose'),
                ('movie_name', 'IT Chapter Two'),
                ('number_of_tickets',
                 '4 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('show_date', '1st of March'), ('show_time', '9:30 pm'),
                ('show_type', '3d of possible values regular, 3d, imax'),
                ('theater_name', 'AMC Saratoga')
            ])),
        Prompt(
            utt="[user] I could do with going out to watch a sports movie. Any good ones in Mountain View? [system] There's 83 playing in Century 16 [user] No foreign language cinema for me. Anything in Palo Alto then? [system] You're in luck, they're playing Moneyball at CineArts. [user] I'll take that. Any shows today? [system] They only have a 7:00 pm, regular show. Should I book you? [user] Yes, one ticket would be nice.",
            slots=OrderedDict([
                ('genre', 'sports'), ('location', 'Palo Alto'),
                ('movie_name', 'Moneyball'),
                ('number_of_tickets',
                 '1 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('show_date', 'today'), ('show_time', '7:00 pm'),
                ('show_type', 'regular of possible values regular, 3d, imax'),
                ('theater_name', 'CineArts')
            ])),
        Prompt(
            utt="[user] Hey can you find tickets for The Matrix Resurrections, the sci-fi movie that just came out? [system] Which city to find a cinema in? [user] Try Mountain View [system] And for what date? [user] December 24 [system] Century 16 has shows throughout the day. There's 9 am, 3 pm and 5 pm. [user] I'll need 6 seats for the 5 pm show [system] Sorry, only the 9 am show can accommodate 6 people. [user] Okay that'll do then. Can you book a 3D show?",
            slots=OrderedDict([
                ('genre', 'sci-fi'), ('location', 'Mountain View'),
                ('movie_name', 'The Matrix Resurrections'),
                ('number_of_tickets',
                 '6 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('show_date', 'December 24'), ('show_time', '9 am'),
                ('show_type', '3d of possible values regular, 3d, imax'),
                ('theater_name', 'Century 16')
            ])),
        Prompt(
            utt="[user] What's playing at AMC Mercado in Santa Clara today? [system] Any genre in mind? [user] Some horror would be lovely [system] Umm, they have The Black Phone that came out recently [user] That's promising, is that available in imax? [system] It actually is! Do you want the 4 pm show? [user] Sure, can you find 2 tickets?",
            slots=OrderedDict([
                ('genre', 'horror'), ('location', 'Santa Clara'),
                ('movie_name', 'The Black Phone'),
                ('number_of_tickets',
                 '2 of possible values 1, 2, 3, 4, 5, 6, 7, 8, 9'),
                ('show_date', 'today'), ('show_time', '4 pm'),
                ('show_type', 'imax of possible values regular, 3d, imax'),
                ('theater_name', 'AMC Mercado')
            ]))
    ],
    'Movies_2': [
        Prompt(
            utt="[user] Any movies I can watch in the Comic genre? [system] I found 10 movies you might like. How about A Madea Family Funeral? The movie has a rating of 4.2. [user] Any others? I'm looking for something directed by Jonathan Levine, and starring O'Shea Jackson Jr. [system] What do you think about Long Shot? It has an average rating of 6.6. [user] Sounds perfect to me.",
            slots=OrderedDict([('genre', 'Comic'),
                               ('director', 'Jonathan Levine'),
                               ('starring', "O'Shea Jackson Jr.")])),
        Prompt(
            utt="[user] I'm in the mood for a funny movie. Anything directed by morton dacosta and/or starring rand harper? [system] Dacosta didn't direct many movies, there's just Auntie Mame that meets the criterion. [user] That's the one. I'd like to watch it.",
            slots=OrderedDict([('director', 'morton dacosta'),
                               ('genre', 'funny'),
                               ('starring', 'rand harper')])),
        Prompt(
            utt="[user] Can you find a drama film for me? [system] I found 10 movies for you. There's Dogman, rated 7.3. [user] Anything else? I want something directed by Robin Bissell and starring Sam Rockwell. [system] There's The Best of Enemies, rated 7.0. [user] Sure.",
            slots=OrderedDict([('director', 'Robin Bissell'),
                               ('genre', 'drama'),
                               ('starring', 'Sam Rockwell')])),
        Prompt(
            utt="[user] Do you know any good Harry Elfont comic flicks? Something with Corinne Reilly would be great. [system] Here's 1. Josie and the Pussycats. It comes in with a 5.4 rating. [user] Nice.",
            slots=OrderedDict([('director', 'Harry Elfont'), ('genre', 'comic'),
                               ('starring', 'Corinne Reilly')])),
        Prompt(
            utt='[user] Hi there, are there any nice Comic movies I can watch which is directed my Harry Elfont and stars Carson Daly? [system] Yes there is 1 movie called Josie and the Pussycats with a 5.4 rating. [user] That sounds great!',
            slots=OrderedDict([('director', 'Harry Elfont'), ('genre', 'Comic'),
                               ('starring', 'Carson Daly')]))
    ],
    'Movies_3': [
        Prompt(
            utt="[user] I want to watch a McKinley Belcher III drama directed by Simon Curtis. [system] I found 1 movie. The Art of Racing in the Rain has a 43 percent rating. [user] That's a good one.",
            slots=OrderedDict([('cast', 'McKinley Belcher III'),
                               ('directed_by', 'Simon Curtis'),
                               ('genre', 'drama')])),
        Prompt(
            utt="[user] I'm looking to watch an action movie directed by David Leitch starring Alex King. [system] How about Hobbs and Shaw? It has an average rating of 67 percent. [user] I quite like that one.",
            slots=OrderedDict([('cast', 'Alex King'),
                               ('directed_by', 'David Leitch'),
                               ('genre', 'action')])),
        Prompt(
            utt='[user] I wanna see a thriller movie, maybe something directed with Riley Stearns and starring Steve Terada [system] How about The Art of Self-Defense which has an 83 rating, you might like that [user] That sounds good then, see you later.',
            slots=OrderedDict([('cast', 'Steve Terada'),
                               ('directed_by', 'Riley Stearns'),
                               ('genre', 'thriller')])),
        Prompt(
            utt="[user] I'd like to watch a Cartoon movie directed by Josh Cooley, preferably with the actor Bill Hader. [system] There's 1 movie: Toy Story 4, with a rating of 97 out of 100 [user] Perfect, thanks!",
            slots=OrderedDict([('cast', 'Bill Hader'),
                               ('directed_by', 'Josh Cooley'),
                               ('genre', 'Cartoon')])),
        Prompt(
            utt='[user] I am looking for Drama movies directed by Gregory La Cava and with Franklin Pangborn acting in it. [system] I found one with a 95 average rating out of a 100 called Stage Door. [user] That sounds great.',
            slots=OrderedDict([('cast', 'Franklin Pangborn'),
                               ('directed_by', 'Gregory La Cava'),
                               ('genre', 'Drama')]))
    ],
    'Music_1': [
        Prompt(
            utt="[user] I want to listen to a good song. Do you have something by Alessia Cara from 2018? [system] Out of Love by Alessia Cara from her album The Pains of Growing. How's that? [user] Something else in Pop from that album? [system] What about Trust My Lonely; also from the album The Pains of Growing? [user] Yes, that's good. [system] Should I play the song? [user] Yes, play it on my kitchen speaker please.",
            slots=OrderedDict([
                ('artist', 'Alessia Cara'),
                ('year',
                 '2018 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                ), ('album', 'The Pains of Growing'), ('genre', 'Pop'),
                ('song_name', 'Trust My Lonely'),
                ('playback_device',
                 'Kitchen speaker of possible values TV, Kitchen speaker, Bedroom speaker'
                )
            ])),
        Prompt(
            utt='[user] I want to listen to a 2014 reggae track from sammielz. [system] would you like to listen to let us be by sammielz from the album the beginning? [user] Is there anything else? Something from rebelution from the album count me in [system] I have found 1 song. You want to hear roots reggae music by rebelution from the album count me in? [user] That would be great. Play it on my bedroom speaker.',
            slots=OrderedDict([
                ('album', 'count me in'), ('artist', 'rebelution'),
                ('genre', 'reggae'),
                ('playback_device',
                 'Bedroom speaker of possible values TV, Kitchen speaker, Bedroom speaker'
                ), ('song_name', 'roots reggae music'),
                ('year',
                 '2014 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ])),
        Prompt(
            utt="[user] Can you search for a Metal song from the album Infestissumam, it's from 2013 I think [system] Found the 1 which is the song Year Zero by Ghost in the album Infestissumam. What about your opinion? [user] Yes, I love to hear that one and play it on my bedroom speaker. [system] Need your confirmation to play the song Year Zero on Bedroom speaker. [user] Sorry, please play it on my TV.",
            slots=OrderedDict([
                ('album', 'Infestissumam'), ('artist', 'Ghost'),
                ('genre', 'Metal'),
                ('playback_device',
                 'TV of possible values TV, Kitchen speaker, Bedroom speaker'),
                ('song_name', 'Year Zero'),
                ('year',
                 '2013 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ])),
        Prompt(
            utt='[user] Can you play me a Pop song from three years back, so 2016? [system] Want to listen to something by RHCP or maybe Shawn Mendes? [user] RHCP, yes! From their album The Getaway maybe? [system] You might like Dark Necessities from that album [user] That one is good. Can you play it now on my TV please?night!',
            slots=OrderedDict([
                ('album', 'The Getaway'), ('artist', 'RHCP'), ('genre', 'Pop'),
                ('playback_device',
                 'TV of possible values TV, Kitchen speaker, Bedroom speaker'),
                ('song_name', 'Dark Necessities'),
                ('year',
                 '2016 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ])),
        Prompt(
            utt='[user] I wish to hear some Pop song by Shawn Mendes. [system] Sure. How about Aftertaste by Shawn Mendes from the album Handwritten? [user] so, when was the release of the song? [system] 2015. [user] Okay! Something from him from 2016, maybe from the album Illuminate? [system] Sure! Do you wish to listen the song Bad Reputation by Shawn Mendes from the album Illuminate? [user] Yup, play it in the bedroom please.',
            slots=OrderedDict([
                ('album', 'Illuminate'), ('artist', 'Shawn Mendes'),
                ('genre', 'Pop'),
                ('playback_device',
                 'Bedroom speaker of possible values TV, Kitchen speaker, Bedroom speaker'
                ), ('song_name', 'Bad Reputation'),
                ('year',
                 '2016 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ]))
    ],
    'Music_2': [
        Prompt(
            utt='[user] Can you play a Soundtracks genre track, album no bar? I like Hiroyuki Sawano. [system] I found 1 song that fits: Perfect Time from the album The Seven Deadly Sins Original Soundtrack. Does that sound good? [user] That works for me, can you play that from my bedroom speaker?',
            slots=OrderedDict([
                ('genre', 'Soundtracks'), ('album', 'dontcare'),
                ('artist', 'Hiroyuki Sawano'), ('song_name', 'Perfect Time'),
                ('playback_device',
                 'bedroom speaker of possible values TV, kitchen speaker, bedroom speaker'
                )
            ])),
        Prompt(
            utt="[user] Will you find me Pop songs? Say Jai Waetford's songs from the album Heart Miles? [system] I found 1 song. Do you like Shy by Jai Waetford from the album Heart Miles? [user] Sounds good. Please play the song on kitchen speaker.",
            slots=OrderedDict([
                ('album', 'Heart Miles'), ('artist', 'Jai Waetford'),
                ('genre', 'Pop'),
                ('playback_device',
                 'kitchen speaker of possible values TV, kitchen speaker, bedroom speaker'
                ), ('song_name', 'Shy')
            ])),
        Prompt(
            utt='[user] Find me some pop songs of Courier from the album The Present tense [system] How about San Francisco by Courier from the album The Present tense [user] That is good [system] should i play it? [user] yes please on the TV',
            slots=OrderedDict([
                ('album', 'The Present tense'), ('artist', 'Courier'),
                ('genre', 'pop'),
                ('playback_device',
                 'TV of possible values TV, kitchen speaker, bedroom speaker'),
                ('song_name', 'San Francisco')
            ])),
        Prompt(
            utt='[user] Can you find me a Halsey song? [system] I found 5, how about Colors on Badlands by Halsey? [user] What else do you have from Badlands? I like electropop. [system] How about Control from Badlands by Halsey? [user] Anything else you have? [system] Gasoline on Badlands by Halsey. [user] Perfect, I want that one. [system] Gasoline cast to the tv is that correct? [user] Yes.',
            slots=OrderedDict([
                ('album', 'Badlands'), ('artist', 'Halsey'),
                ('genre', 'electropop'),
                ('playback_device',
                 'TV of possible values TV, kitchen speaker, bedroom speaker'),
                ('song_name', 'Gasoline')
            ])),
        Prompt(
            utt="[user] Can you find some Country songs in the album Experiment? [system] Yes, that's a nice album by Kane Brown. There are 3 songs you may enjoy: any of Homesick, Lose It or Weekend appeal to you? [user] I like Weekend!! [system] Should I play the music? [user] Yes, please do so on my TV.",
            slots=OrderedDict([
                ('album', 'Experiment'), ('artist', 'Kane Brown'),
                ('genre', 'Country'),
                ('playback_device',
                 'TV of possible values TV, kitchen speaker, bedroom speaker'),
                ('song_name', 'Weekend')
            ]))
    ],
    'Music_3': [
        Prompt(
            utt='[user] Can you find me a pop song by Meghan Trainor from 2014? [system] How about All about That Bass by Meghan Trainor from the album Title? Can you search again for songs by Redfoo from the album Party Rock Mansion? [system] There is 1 song called New Thang by Redfoo from the album Party Rock Mansion. [user] Yes, That is the song I am looking for. Wanna listen to it in the patio... err no my living room.',
            slots=OrderedDict([
                ('artist', 'Redfoo'), ('genre', 'pop'),
                ('year',
                 '2014 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                ), ('album', 'Party Rock Mansion'), ('track', 'New Thang'),
                ('device',
                 'Living room of possible values Living room, Kitchen, Patio')
            ])),
        Prompt(
            utt="[user] I'd like to find some songs now, hopefully some pop from Madonna from '19. [system] I have 2 songs to recommend. Maybe Batuka, or God Control from her album Madame X. [user] That works, thank you. [system] Do you want to hear that now? [user] Yes, I do, in my living room please.",
            slots=OrderedDict([
                ('album', 'Madame X'), ('artist', 'Madonna'),
                ('device',
                 'Living room of possible values Living room, Kitchen, Patio'),
                ('genre', 'pop'), ('track', 'God Control'),
                ('year',
                 '2019 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ])),
        Prompt(
            utt='[user] Hey I could listen to some pop rock from Coldplay. Anything from their 2015 album? [system] A Head Full Of Dreams was released that year. You might like Everglow from that album. Want me to play it? [user] Sure, play that on my TV... er no my kitchen speaker if you can.',
            slots=OrderedDict([
                ('album', 'A Head Full Of Dreams'), ('artist', 'Coldplay'),
                ('device',
                 'Kitchen of possible values Living room, Kitchen, Patio'),
                ('genre', 'pop rock'), ('track', 'Everglow'),
                ('year',
                 '2015 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ])),
        Prompt(
            utt="[user] I'm so angry, play some metal for me. [system] Sure, any song or artist you want me to search by? [user] I listen to Slipknot a lot, try them. [system] Okay, Unsainted from their 2019 album We Are Not Your Kind might be suited for your mood right now. [user] I haven't heard it, so let's try it on my patio speaker!",
            slots=OrderedDict([
                ('album', 'We Are Not Your Kind'), ('artist', 'Slipknot'),
                ('device',
                 'Patio of possible values Living room, Kitchen, Patio'),
                ('genre', 'metal'), ('track', 'Unsainted'),
                ('year',
                 '2019 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ])),
        Prompt(
            utt="[user] I need to prepare a popular song for a singing recital. Can you find me some popular song in the Pop genre from 2012? [system] There's a few, We Are Young by Fun and Call Me Maybe by Carly Rae Jepsen were big then [user] The latter sounds perfect for my recital. What album is it from? [system] The album is called Kiss. [user] Perfect, I wanna play that song in the living room.",
            slots=OrderedDict([
                ('album', 'Kiss'), ('artist', 'Carly Rae Jepsen'),
                ('device',
                 'Living room of possible values Living room, Kitchen, Patio'),
                ('genre', 'Pop'), ('track', 'Call Me Maybe'),
                ('year',
                 '2012 of possible values 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019'
                )
            ]))
    ],
    'Payment_1': [
        Prompt(
            utt='[user] I want to make a payment to Margaret. [system] How much funds and from which account do you want me transfer? [user] I would like to transfer 33 dollars from visa. [system] Please confirm: You want me to make a transfer of $33 from your credit card to Margaret in public. [user] Yes, That is correct.',
            slots=OrderedDict([(
                'payment_method',
                'credit card of possible values app balance, debit card, credit card'
            ), ('private_visibility', 'False of possible values True, False'),
                               ('receiver', 'Margaret'), ('amount', '$33')])),
        Prompt(
            utt="[user] I need to pay someone. [system] From which account, to whom, and how much? [user] I want to pay Mary eighty five bucks via MasterCard. [system] Fine, here's what I heard: Pay Mary $85 from credit card? Should I hide the transaction? [user] That's right, yes. No need to hide it though.",
            slots=OrderedDict([
                ('amount', '$85'),
                ('payment_method',
                 'credit card of possible values app balance, debit card, credit card'
                ),
                ('private_visibility', 'False of possible values True, False'),
                ('receiver', 'Mary')
            ])),
        Prompt(
            utt="[user] I want to make a payment to Jerry for $82 from my mastercard [system] Confirming you want to pay Jerry $82 with your credit card yes? [user] Yes that's right, make the transaction private too",
            slots=OrderedDict([
                ('amount', '$82'),
                ('payment_method',
                 'credit card of possible values app balance, debit card, credit card'
                ),
                ('private_visibility', 'True of possible values True, False'),
                ('receiver', 'Jerry')
            ])),
        Prompt(
            utt="[user] Please make a $93 payment from my money in app. [system] Please say the contact's name. [user] The contact's name is Mary. Please send it from my credit card instead. [system] Okay. Please confirm that you'd like to make a publicly visible, $93 credit card payment to Mary. [user] You got that right.",
            slots=OrderedDict([
                ('amount', '$93'),
                ('payment_method',
                 'credit card of possible values app balance, debit card, credit card'
                ),
                ('private_visibility', 'False of possible values True, False'),
                ('receiver', 'Mary')
            ])),
        Prompt(
            utt="[user] I want to make a payment to Mary. [system] Do you want to transfer the money from app balance or debit card? [user] Send 105 bucks from my UnionPay, please. [system] Please confirm: Sending 105 bucks to Mary from your credit card. This transaction is private. [user] Yes, that's correct.",
            slots=OrderedDict([
                ('amount', '105 bucks'),
                ('payment_method',
                 'credit card of possible values app balance, debit card, credit card'
                ),
                ('private_visibility', 'True of possible values True, False'),
                ('receiver', 'Mary')
            ]))
    ],
    'RentalCars_1': [
        Prompt(
            utt="[user] Book me a rental car. [system] Sure. What time will you pick it up? What date will you return the car? [user] I'll pick it up around 10:30 in the morning, keeping it until next Thursday [system] Where will you be picking it up? What day? [user] I'll pick it up in New York on March 4th [system] I have 6 cars for you. There's a standard car, an Altima, available at Port Authority Station next Monday [user] Sounds good",
            slots=OrderedDict([
                ('dropoff_date', 'next Thursday'),
                ('pickup_time', '10:30 in the morning'),
                ('pickup_city', 'New York'), ('pickup_date', 'March 4th'),
                ('pickup_location', 'Port Authority Station'),
                ('type',
                 'Standard of possible values Compact, Standard, Full-size')
            ])),
        Prompt(
            utt="[user] find a rental car [system] where do you want it? [user] starting march 8th in san francisco, a standard sedan will do [system] till when do you want it? [user] till 11th of this month. I'd like to pick it up at three in the afternoon [system] I found 6 cars at sfo international airport. Would you like an Altima? [user] that is good, can you rserve that?",
            slots=OrderedDict([
                ('dropoff_date', '11th of this month'),
                ('pickup_city', 'san francisco'), ('pickup_date', 'march 8th'),
                ('pickup_location', 'sfo international airport'),
                ('pickup_time', '3 pm'),
                ('type',
                 'Standard of possible values Compact, Standard, Full-size')
            ])),
        Prompt(
            utt="[user] I need a rental car. [system] What are your rental dates and where pick up? [user] Pick up is Phoenix and rental is March 10th to the 11th. [system] What time is pickup? [user] I want to pick up 9 am. [system] At Sky Harbor International Airport, I have a Standard Accord for pick up March 10th. [user] That is perfect for me. What is the cost? I'd like to rent.",
            slots=OrderedDict([
                ('dropoff_date', 'March 11th'), ('pickup_city', 'Phoenix'),
                ('pickup_date', 'March 10th'),
                ('pickup_location', 'Sky Harbor International Airport'),
                ('pickup_time', '9 am'),
                ('type',
                 'Standard of possible values Compact, Standard, Full-size')
            ])),
        Prompt(
            utt="[user] I need a rental car. [system] Where do you want to pick up the rental car? How long do you want the rental car for? [user] I want to pick up the rental car from Phoenix until the 7th. [system] What dates do you want the rental car? [user] I would like the rental car from March 6th at around evening 6. [system] There's a Standard Accord for rent available next Wednesday at Sky Harbor International Airport. [user] Sounds good!",
            slots=OrderedDict([
                ('dropoff_date', 'the 7th'), ('pickup_city', 'Phoenix'),
                ('pickup_date', 'March 6th'),
                ('pickup_location', 'Sky Harbor International Airport'),
                ('pickup_time', 'evening 6'),
                ('type',
                 'Standard of possible values Compact, Standard, Full-size')
            ])),
        Prompt(
            utt='[user] I need a rental car until 11th of March in London, UK. [system] What time? When? [user] I need the car from the 5th and I want to pick it up at 3 in the afternoon. [system] There is a car at Heathrow International Airport for next Tuesday. It is a Standard Altima. There are 4 car options. [user] Is there anything else? [system] There is a Standard car, a Cruze, available at Heathrow International Airport for pickup the 5th. [user] That will work.',
            slots=OrderedDict([
                ('dropoff_date', '11th of March'),
                ('pickup_city', 'London, UK'), ('pickup_date', 'next Tuesday'),
                ('pickup_location', 'Heathrow International Airport'),
                ('pickup_time', '3 in the afternoon'),
                ('type',
                 'Standard of possible values Compact, Standard, Full-size')
            ]))
    ],
    'RentalCars_2': [
        Prompt(
            utt='[user] I am looking to rent a car, and I would need to pick it up at 09:30. [system] How long will you be needing the rental? [user] I will be needing the car till the 13th. [system] When and where you you like to pick up the rental car? [user] I would like to pick up the car in CDMX and I will need it on the 10th of March. [system] Okay, I have a standard style Chevrolet Cruze available for pick up March 10th in Estacion Buenavista. Would that work? [user] That would work just fine.',
            slots=OrderedDict([
                ('pickup_time', '09:30'), ('dropoff_date', 'March 13th'),
                ('pickup_city', 'CDMX'), ('pickup_date', '10th of March'),
                ('car_type',
                 'Standard of possible values Compact, Standard, Full-size'),
                ('pickup_location', 'Estacion Buenavista')
            ])),
        Prompt(
            utt='[user] Can you help me get a rental car in Vancouver, BC starting March 12th? I need the car until the 14th. [system] What time of day do you need it? [user] 2 in the afternoon and I want an intermediate car. [system] If you pick up at Pacific Central Station on March 12th you can get a standard car, Chevrolet Cruze. [user] I love that option.',
            slots=OrderedDict([
                ('car_type',
                 'Standard of possible values Compact, Standard, Full-size'),
                ('dropoff_date', 'the 14th'), ('pickup_city', 'Vancouver, BC'),
                ('pickup_date', 'March 12th'),
                ('pickup_location', 'Pacific Central Station'),
                ('pickup_time', '2 in the afternoon')
            ])),
        Prompt(
            utt="[user] Hi there, I'd like to find a rental car. [system] What time do you need it by and till what date? When from? [user] I want it till 10th of March and I need it on March 8th at five in the evening. [system] Where will you pick it up? [user] In Philly please. [system] There is a good car available at PHL International Airport which you could pick up next Friday. It's a Standard Chevrolet Cruze [user] Yes that sounds great.",
            slots=OrderedDict([
                ('car_type',
                 'Standard of possible values Compact, Standard, Full-size'),
                ('dropoff_date', '10th of March'), ('pickup_city', 'Philly'),
                ('pickup_date', 'March 8th'),
                ('pickup_location', 'PHL International Airport'),
                ('pickup_time', 'five in the evening')
            ])),
        Prompt(
            utt="[user] I need a rental car until the 9th of this month, can you help? [system] Sure, when would you like to pick it up? [user] On the 7th in New York at afternoon 1:30, thanks. [system] There's a Standard Honda Accord available from next Thursday at Grand Central Station. Is that okay? [user] That'd do, how much is it?",
            slots=OrderedDict([
                ('car_type',
                 'Standard of possible values Compact, Standard, Full-size'),
                ('dropoff_date', 'the 9th'), ('pickup_city', 'New York'),
                ('pickup_date', 'next Thursday'),
                ('pickup_location', 'Grand Central Station'),
                ('pickup_time', 'afternoon 1:30')
            ])),
        Prompt(
            utt="[user] I'll need a rental car in Anaheim, CA on the 9th of March from 3 o'clock in the afternoon, can you book one for me? [system] Great, and until when will you need it? [user] I'll need it until March 10th. [system] I have 6 available cars on March 9th. I'd suggest the Standard Chevrolet Cruze at John Wayne Airport. [user]That sounds good, thanks.",
            slots=OrderedDict([
                ('car_type',
                 'Standard of possible values Compact, Standard, Full-size'),
                ('dropoff_date', 'March 10th'), ('pickup_city', 'Anaheim, CA'),
                ('pickup_date', '9th of March'),
                ('pickup_location', 'John Wayne Airport'),
                ('pickup_time', "3 o'clock in the afternoon")
            ]))
    ],
    'RentalCars_3': [
        Prompt(
            utt="[user] I'd like to look for a rental vehicle in Paris, France. [system] How long do you require the vehicle for, and when would you like to pick it up? [user] I'd like to have the vehicle from half past 6 in the evening to March 5th. [system] What day would you like to begin the vehicle rental? [user] I require the vehicle from the 3rd of this month. [system] There is a lovely Citroen C5 SUV available from Orly International Airport. [user] That's fine with me. [system] Would you like to reserve the vehicle? [user] Yeah, I'd like to rent the car sans insurance.",
            slots=OrderedDict([
                ('city', 'Paris, France'), ('end_date', 'March 5th'),
                ('pickup_time', 'half past 6 in the evening'),
                ('start_date', '3rd of this month'),
                ('car_type', 'SUV of possible values Hatchback, Sedan, SUV'),
                ('pickup_location', 'Orly International Airport'),
                ('add_insurance', 'False of possible values True, False')
            ])),
        Prompt(
            utt='[user] I need to find a rental car March 1st, 18:00. [system] Where are you looking to rent a car from, how long do you need it [user] I want to find a hatchback in Philly. I need it until March 4th [system] There is a hatchback Kia Ceed available at the PHL International Airport. [user] Sounds good. I want to reserve it. [system] Do you want insurance, as well? [user] Yes please.',
            slots=OrderedDict([
                ('add_insurance', 'True of possible values True, False'),
                ('car_type',
                 'Hatchback of possible values Hatchback, Sedan, SUV'),
                ('city', 'Philly'), ('end_date', 'March 4th'),
                ('pickup_location', 'PHL International Airport'),
                ('pickup_time', '18:00'), ('start_date', 'March 1st')
            ])),
        Prompt(
            utt="[user] Find for me a rental car, need to pick up on the 4th 19:30. [system] Do you need the car until and from where? [user] Till March 11th, from Rio. [system] I have for you 1 car available.A nice Hatchback of Kia Ceed available at Galeao International Airport. [user] Sounds very good. [system] You want to book the car? [user] Yes rent it for me please. [system] Insurance you want to add? [user] No.I don't need it.",
            slots=OrderedDict([
                ('add_insurance', 'False of possible values True, False'),
                ('car_type',
                 'Hatchback of possible values Hatchback, Sedan, SUV'),
                ('city', 'Rio'), ('end_date', 'March 11th'),
                ('pickup_location', 'Galeao International Airport'),
                ('pickup_time', '19:30'), ('start_date', 'the 4th')
            ])),
        Prompt(
            utt="[user] I'll be in Sacramento soon and will need a rental car. [system] What time will you pick up the car? [user] I will pick it up at six in the evening on 8th of this month. [system] When do you want to return the car? [user] I will need the car until the 10th. [system] There's a good car at Sacramento Valley Station. It's a Buick Regal Sportback sedan. [user] Sounds great. Reserve the car, skip insurance.",
            slots=OrderedDict([
                ('add_insurance', 'False of possible values True, False'),
                ('car_type', 'Sedan of possible values Hatchback, Sedan, SUV'),
                ('city', 'Sacramento'), ('end_date', 'the 10th'),
                ('pickup_location', 'Sacramento Valley Station'),
                ('pickup_time', '6 in the evening'),
                ('start_date', '8th of this month')
            ])),
        Prompt(
            utt="[user] Can you help me find a car to rent from March 8 through the 12th? [system] What location are we talking about? [user] I'm talking about the Long Beach area [system] I found 10 cars including a SUV, a Alfa Romeo Stelvio that you can pick up from Downtown Station [user] How much is that a day [system] It costs $36.00 a day [user] That sounds good for me then. Can you book it for pickup at 12:30 pm? Don't forget to insure the rental.",
            slots=OrderedDict([
                ('add_insurance', 'True of possible values True, False'),
                ('car_type', 'SUV of possible values Hatchback, Sedan, SUV'),
                ('city', 'Long Beach'), ('end_date', 'the 12th'),
                ('pickup_location', 'Downtown Station'),
                ('pickup_time', '12:30 pm'), ('start_date', '8March 8')
            ]))
    ],
    'Restaurants_1': [
        Prompt(
            utt="[user] I'd like to find somewhere intermediate priced to eat. [system] What kind of food are you interested in, and in which city should I conduct my search? [user] Something Southern with a live band in San Jose would be great. We'll be drinking alcohol. [system] You're gonna love Rosie Mccann's Irish Pub & Restaurant. [user] Sounds great. Make the reservation for 2 for 11:45 am.",
            slots=OrderedDict([(
                'price_range',
                'moderate of possible values inexpensive, moderate, expensive, very expensive'
            ), ('city', 'San Jose'), ('cuisine', 'Southern'),
                               ('has_live_music',
                                'True of possible values True, False'),
                               ('serves_alcohol',
                                'True of possible values True, False'),
                               ('restaurant_name',
                                "Rosie Mccann's Irish Pub & Restaurant"),
                               ('time', '11:45 am'), ('date', 'today'),
                               ('party_size',
                                '2 of possible values 1, 2, 3, 4, 5, 6')])),
        Prompt(
            utt="[user] Can you recommend some moderately priced Pakistani places to eat? I have a craving. Don't bother with alcohol or live music [system] Sure, did you have a city in mind? [user] The best ones are in Sunnyvale, check there [system] Shah is considered pretty good. Want me to book a table there? [user] Absolutely, sign me up for 8pm today if you can, table for one",
            slots=OrderedDict([(
                'price_range',
                'moderate of possible values inexpensive, moderate, expensive, very expensive'
            ), ('city', 'Sunnyvale'), ('cuisine', 'Pakistani'),
                               ('has_live_music',
                                'False of possible values True, False'),
                               ('serves_alcohol',
                                'False of possible values True, False'),
                               ('restaurant_name', 'Shah'), ('time', '8pm'),
                               ('date', 'today'),
                               ('party_size',
                                '1 of possible values 1, 2, 3, 4, 5, 6')])),
        Prompt(
            utt="[user] I have a date coming up, can you help me find a restaurant? [system] What cuisine and city? [user] He would like Portuguese I think. Go for a luxury restarant in San Jose with alcohol, but no music [system] There's literally just one place that matches. How does Adega sound? [user] That'll be good. Book me a table for me and him for 7 pm on the 9th?",
            slots=OrderedDict([(
                'price_range',
                'moderate of possible values inexpensive, moderate, expensive, very expensive'
            ), ('city', 'San Jose'), ('cuisine', 'Portuguese'),
                               ('has_live_music',
                                'False of possible values True, False'),
                               ('serves_alcohol',
                                'True of possible values True, False'),
                               ('restaurant_name', 'Adega'), ('time', '7 pm'),
                               ('date', 'the 9th'),
                               ('party_size',
                                '2 of possible values 1, 2, 3, 4, 5, 6')])),
        Prompt(
            utt="[user] I'm visiting Mexico City. Can you find a mid-ranged Mexican place there? [system] Cafe de Tacuba is supposed to be good. [user] Do they have a live band and alcohol? [system] Yes to both! [user] Fine, book me a table there? [system] For how many and when? [user] I plan on getting there 3 pm on the 18th if you can do that. I'll be alone",
            slots=OrderedDict([(
                'price_range',
                'moderate of possible values inexpensive, moderate, expensive, very expensive'
            ), ('city', 'Mexico City'), ('cuisine', 'Mexican'),
                               ('has_live_music',
                                'True of possible values True, False'),
                               ('serves_alcohol',
                                'True of possible values True, False'),
                               ('restaurant_name', 'Cafe de Tacuba'),
                               ('time', '3 pm'), ('date', '18th'),
                               ('party_size',
                                '1 of possible values 1, 2, 3, 4, 5, 6')])),
        Prompt(
            utt="[user] There's this super cheap Desi place in LA that I want to visit. No alcohol or live music if I remember correctly. Is there something? [system] You're probably referring to Bombay Frankie Company [user] Yes, that one! Do they take reservations? [system] They do, when do you want one and for how many? [user] 11 am on Saturday next week for 2, thanks",
            slots=OrderedDict([(
                'price_range',
                'inexpensive of possible values inexpensive, moderate, expensive, very expensive'
            ), ('city', 'LA'), ('cuisine', 'Desi'),
                               ('has_live_music',
                                'False of possible values True, False'),
                               ('serves_alcohol',
                                'False of possible values True, False'),
                               ('restaurant_name', 'Bombay Frankie Company'),
                               ('time', '11 am'),
                               ('date', 'Saturday next week'),
                               ('party_size',
                                '2 of possible values 1, 2, 3, 4, 5, 6')]))
    ],
    'Restaurants_2': [
        Prompt(
            utt='[user] How about finding a place March 3rd? Somewhere moderate in cost that has vegetarian menu items. [system] I assume in Novato? [user] Novato is correct. Something serving Latin American cuisine and if possible with outdoor seating. [system] I found 1 called Maya Palenque Restaurant. [user] I bet they have good food. [system] Should I book you a table? [user] Yes for two please. [system] What time would you like it for? [user] In the morning 11:15 please.',
            slots=OrderedDict([
                ('category', 'Latin American'),
                ('has_vegetarian_options',
                 'True of possible values True, False'), ('location', 'Novato'),
                ('price_range',
                 'moderate of possible values cheap, moderate, pricey, ultra high-end'
                ),
                ('has_seating_outdoors', 'True of possible values True, False'),
                ('restaurant_name', 'Maya Palenque Restaurant'),
                ('date', 'March 3rd'),
                ('number_of_seats', '2 of possible values 1, 2, 3, 4, 5, 6'),
                ('time', 'morning 11:15')
            ])),
        Prompt(
            utt="[user] I am searching for an inexpensive veggie-friendly restaurant in Morgan Hill with outdoor seating [system] What type of food do you need? [user] I wish to have Pizza and Pasta. [system] I searched and found 3 restaurant for you. Betto's Bistro is a great restaurant in Morgan Hill. [user] Great, can you reserve them for March 1st? Say 12:15 in the afternoon?",
            slots=OrderedDict([
                ('category', 'Pizza and Pasta'), ('date', 'March 1st'),
                ('has_seating_outdoors', 'True of possible values True, False'),
                ('has_vegetarian_options',
                 'True of possible values True, False'),
                ('location', 'Morgan Hill'),
                ('number_of_seats', '2 of possible values 1, 2, 3, 4, 5, 6'),
                ('price_range',
                 'cheap of possible values cheap, moderate, pricey, ultra high-end'
                ), ('restaurant_name', "Betto's Bistro"),
                ('time', '12:15 in the afternoon')
            ])),
        Prompt(
            utt="[user] I need to find a pricey place to eat. Spanish cuisine ideally, in Mountain View. Don't bother with vegetarian options, but we'd like to sit outside. [system] There's Cascal, which is renowned. [user] Yes, that one sounds good. Can you book a table? [system] For how many and when? [user] tomorrow at 5 pm, for 6. [system] They can't accommodate that many people at that time, should I try another time? [user] Can you try 6 pm?",
            slots=OrderedDict([
                ('category', 'Spanish'), ('date', 'tomorrow'),
                ('has_seating_outdoors', 'True of possible values True, False'),
                ('has_vegetarian_options',
                 'False of possible values True, False'),
                ('location', 'Mountain View'),
                ('number_of_seats', '6 of possible values 1, 2, 3, 4, 5, 6'),
                ('price_range',
                 'pricey of possible values cheap, moderate, pricey, ultra high-end'
                ), ('restaurant_name', 'Cascal'), ('time', '6 pm')
            ])),
        Prompt(
            utt="[user] Hey can y ou find me a high-end steakhouse kind of restaurant? No veggie options needed [system] Which city in? [user] Say, San Francisco [system] Would you like to try Fogo de Chao? It's a nice Brazilian steakhouse [user] That sounds fun. Do they have seating outdoors? [system] They do [user] Make a reservation if you can. 4 people on the 15th of March [system] What time would you like the reservation for? They open at 6 pm [user] 7 pm then.",
            slots=OrderedDict([
                ('category', 'steakhouse'), ('date', '15th of March'),
                ('has_seating_outdoors', 'True of possible values True, False'),
                ('has_vegetarian_options',
                 'False of possible values True, False'),
                ('location', 'San Francisco'),
                ('number_of_seats', '4 of possible values 1, 2, 3, 4, 5, 6'),
                ('price_range',
                 'pricey of possible values cheap, moderate, pricey, ultra high-end'
                ), ('restaurant_name', 'Fogo de Chao'), ('time', '7 pm')
            ])),
        Prompt(
            utt='[user] My parents from Thailand are visiting and they would like Thai food. Can you find a place? [system] Cool, where? [user] I want it in Palo Alto [system] Thaiphoon is a moderate priced restaurant there, there is also Lotus Thai Bistro which is cheap. [user] Thaiphoon would be good. Do they have vegetarian options and outdoor seating? [system] Yes to both [user] Perfect, find a table there for 3 then on Sunday for 5 pm',
            slots=OrderedDict([
                ('category', 'Thai'), ('date', 'Sunday'),
                ('has_seating_outdoors', 'True of possible values True, False'),
                ('has_vegetarian_options',
                 'True of possible values True, False'),
                ('location', 'Palo Alto'),
                ('number_of_seats', '3 of possible values 1, 2, 3, 4, 5, 6'),
                ('price_range',
                 'moderate of possible values cheap, moderate, pricey, ultra high-end'
                ), ('restaurant_name', 'Thaiphoon'), ('time', '5 pm')
            ]))
    ],
    'RideSharing_1': [
        Prompt(
            utt='[user] I missed my train will you call me a Shared cab ride? [system] Please tell the number of passengers ? [user] Just myself [system] To which place ride should go to? [user] Ride need to go to 3090 Olsen Drive.',
            slots=OrderedDict([
                ('shared_ride', 'True of possible values True, False'),
                ('number_of_riders', '1 of possible values 1, 2, 3, 4'),
                ('destination', '3090 Olsen Drive')
            ])),
        Prompt(
            utt="[user] Call me a cab please. Going to Hop Creek Pub. It can be a shared ride. [system] How many people are going? [user] Just one [system] Please confirm: a shared ride to Hop Creek Pub for 1 person? [user] Yes that's right. How much is it?",
            slots=OrderedDict([
                ('destination', 'Hop Creek Pub'),
                ('number_of_riders', '1 of possible values 1, 2, 3, 4'),
                ('shared_ride', 'True of possible values True, False')
            ])),
        Prompt(
            utt="[user] Will you call a cab for 1 person? [system] You want a shared ride? [user] That's fine. [system] Where to? [user] Cables Wake Park. [system] Please confirm a shared ride for 1 person to Cables Wake Park. [user] No, make that 2 people.",
            slots=OrderedDict([
                ('destination', 'Cables Wake Park'),
                ('number_of_riders', '2 of possible values 1, 2, 3, 4'),
                ('shared_ride', 'True of possible values True, False')
            ])),
        Prompt(
            utt="[user] Can you call a cab? [system] Where would you like to go? [user] I need to go to Arbol Verde Apartments. [system] How many people will be going? [user] It's for 1 person. [system] Do you want a shared ride? [user] I a ok with a ride share service.",
            slots=OrderedDict([
                ('destination', 'Arbol Verde Apartments'),
                ('number_of_riders', '1 of possible values 1, 2, 3, 4'),
                ('shared_ride', 'True of possible values True, False')
            ])),
        Prompt(
            utt="[user] What do I have schedule today? [system] You have a 12 pm property viewing at 200 Greenbriar Circle [user] Great, can you book a cab to that address for 4 people? We don't wanna share the ride",
            slots=OrderedDict([
                ('destination', '200 Greenbriar Circle'),
                ('number_of_riders', '4 of possible values 1, 2, 3, 4'),
                ('shared_ride', 'False of possible values True, False')
            ]))
    ],
    'RideSharing_2': [
        Prompt(
            utt='[user] I would like book a cab ride to 659 Merchant Street. [system] Is there a specific type of ride you would prefer? You can ride Pool, Luxury, or there are other options. [user] I would like to have a regular ride. [system] How many are in your group? [user] Me and my wife, so two.',
            slots=OrderedDict([
                ('destination', '659 Merchant Street'),
                ('ride_type',
                 'Regular of possible values Pool, Regular, Luxury'),
                ('number_of_seats', '2 of possible values 1, 2, 3, 4')
            ])),
        Prompt(
            utt="[user] I'm looking for a cab for one person. [system] Preferred ride type? [user] Luxury, please. [system] Where to? [user] I'm going to Greentree Apartments.",
            slots=OrderedDict([
                ('destination', 'Greentree Apartments'),
                ('number_of_seats', '1 of possible values 1, 2, 3, 4'),
                ('ride_type', 'Luxury of possible values Pool, Regular, Luxury')
            ])),
        Prompt(
            utt='[user] I need a luxury cab for three. [system] Where are you going? [user] I need to get to Loft Bar & Bistro',
            slots=OrderedDict([
                ('destination', 'Loft Bar & Bistro'),
                ('number_of_seats', '3 of possible values 1, 2, 3, 4'),
                ('ride_type', 'Luxury of possible values Pool, Regular, Luxury')
            ])),
        Prompt(
            utt='[user] Will you check events on my calendar? [system] Which date? [user] 10th of March. [system] Stylist appointment at 5:30 pm. [user] Great. Will you reserve a cab there? [system] What type of ride to 2327 Blanding Avenue STE D? [user] Luxury. [system] How many seats? [user] 1.',
            slots=OrderedDict([
                ('destination', '2327 Blanding Avenue STE D'),
                ('number_of_seats', '1 of possible values 1, 2, 3, 4'),
                ('ride_type', 'Luxury of possible values Pool, Regular, Luxury')
            ])),
        Prompt(
            utt='[user] I need a cab. [system] Where do you want to go and for how many people. [user] three people and I want to ride Go Vegan. [system] What type of ride do you prefer? [user] A Pool ride for 2 people instead',
            slots=OrderedDict([
                ('destination', 'Go Vegan'),
                ('number_of_seats', '2 of possible values 1, 2, 3, 4'),
                ('ride_type', 'Pool of possible values Pool, Regular, Luxury')
            ]))
    ],
    'Services_1': [
        Prompt(
            utt="[user] I want to find a unisex salon somewhere in Concord [system] I found 6. First Class Barber Shop is one nice salon. [user] That place sounds good. I'll make an appointment for that place at 18:30. [system] What date will your appointment be? [user] I will visit the place on March 13th.",
            slots=OrderedDict([
                ('is_unisex', 'True of possible values True, False'),
                ('city', 'Concord'), ('appointment_time', '18:30'),
                ('stylist_name', 'First Class Barber Shop'),
                ('appointment_date', 'March 13th')
            ])),
        Prompt(
            utt='[user] I am looking for a Salon [system] In which city are you looking? [user] Martinez, and it gotta be unisex [system] I found 2 salons in Martinez, one is Citrus Salon. [user] Yes that sounds good. [system] Do you want me to book? [user] Yes please go ahead and book. [system] At what time would you like? [user] At 4pm next Monday if you can.',
            slots=OrderedDict([('appointment_date', 'next Monday'),
                               ('appointment_time', '4 pm'),
                               ('city', 'Martinez'),
                               ('is_unisex',
                                'True of possible values True, False'),
                               ('stylist_name', 'Citrus Salon')])),
        Prompt(
            utt="[user] I want a unisex salon in Corte Madera, really need a haircut. [system] Colour Bar is a salon in Corte Madera. [user] Yeah that's good. I want an appointment at this salon. [system] What time and date do you want the appointment to be? [user] I want the appointment to be at five pm on March 11th.",
            slots=OrderedDict([('appointment_date', 'March 11th'),
                               ('appointment_time', '5 pm'),
                               ('city', 'Corte Madera'),
                               ('is_unisex',
                                'True of possible values True, False'),
                               ('stylist_name', 'Colour Bar')])),
        Prompt(
            utt='[user] I want to do a haircut in women-only Salon. Can you find a salon in San Rafael. [system] I have found 2 salons which you may like. The best solon in San Rafael is Best Coast Cuts [user] It sounds good. I want to book an appointment on 9th of March at 17:30.',
            slots=OrderedDict([('appointment_date', '9th of March'),
                               ('appointment_time', '17:30'),
                               ('city', 'San Rafael'),
                               ('is_unisex',
                                'False of possible values True, False'),
                               ('stylist_name', 'Best Coast Cuts')])),
        Prompt(
            utt='[user] I want to find a unisex salon in Berkeley. [system] I found 10 salons in Berkeley. Berkeley Hair Studio is one of them. [user] What is the rating and contact number? [system] The phone number is 510-540-1261. The rating in 4.8. [user] I would like to book an appointment for half past 10 in the morning on the 10th of this month.',
            slots=OrderedDict([
                ('appointment_date', '10th of this month'),
                ('appointment_time', 'half past 10 in the morning'),
                ('city', 'Berkeley'),
                ('is_unisex', 'True of possible values True, False'),
                ('stylist_name', 'Berkeley Hair Studio')
            ]))
    ],
    'Services_2': [
        Prompt(
            utt='[user] Are there any dentists in Sunnyvale, because I really need to find one! [system] There are at least 10 dentists in Sunnyvale; what about Amazing Smile Dental? [user] No, I need a cosmetic dentist closer to Santa Rosa. [system] There is the office of Krystle Lim in Santa Rosa. [user] I find this agreeable. [system] What else may I accomplish for you? [user] For the appointment with Krystle Lim; could you please set one for half past 4 in the evening this Sunday? [system] What day do you want the appointment? [user] Please book the appointment for the 13th at 10:30 in the morning instead.',
            slots=OrderedDict([('city', 'Santa Rosa'),
                               ('offers_cosmetic_services',
                                'True of possible values True, False'),
                               ('dentist_name', 'Krystle Lim'),
                               ('appointment_time', '10:30 in the morning'),
                               ('appointment_date', 'the 13th')])),
        Prompt(
            utt="[user] Can you find me a dentist that offers cosmetic services? [system] What city would you like me to search in? [user] I'd like something in Mountain View [system] I found 2 dentists. They are at Grant Road Dental in Mountain View. [user] I would like to schedule a visit with this dentist. I think I can make the 8th of this month in the afternoon 2:45",
            slots=OrderedDict([('appointment_date', '8th of this month'),
                               ('appointment_time', 'afternoon 2:45'),
                               ('city', 'Mountain View'),
                               ('dentist_name', 'Grant Road Dental'),
                               ('offers_cosmetic_services',
                                'True of possible values True, False')])),
        Prompt(
            utt="[user] Please help me find a dentist in San Jose. Avoid places offering cosmetic services, they tend to be expensive [system] How about Dr. Svetlana Naret? [user] Yes, that one sounds good. What's the weather forecast for March 6th? [system] The average temperature will be 87 degrees. [user] Okay. I would like to make an appointment then at 11 am [system] Sorry, I was unable to book that exact time. How about March 6th at 11:30 am instead? This is still with Dr. Svetlana Naret. [user] Yes, that's good.",
            slots=OrderedDict([('appointment_date', 'March 6th'),
                               ('appointment_time', '11:30 am'),
                               ('city', 'San Jose'),
                               ('dentist_name', 'Dr. Svetlana Naret'),
                               ('offers_cosmetic_services',
                                'False of possible values True, False')])),
        Prompt(
            utt='[user] Can you find a dentist for cosmetic services in Los Altos? [system] There is 1 in Los Altos called Family Dentistry - Dr. Marina Manosov. [user] What is the address? [system] 4646 El Camino Real. [user] I would like an appointment at 4:30 in the evening, 9th of March.',
            slots=OrderedDict([('appointment_date', '9th of March'),
                               ('appointment_time', '4:30 in the evening'),
                               ('city', 'Los Altos'),
                               ('dentist_name',
                                'Family Dentistry - Dr. Marina Manosov'),
                               ('offers_cosmetic_services',
                                'True of possible values True, False')])),
        Prompt(
            utt='[user] Can you find a dentist for me? [system] Which city should I search in? [user] I am looking for one that offers cosmetic services in San Ramon. [system] I found 3. How about Perfect Smiles Family Dentistry located in San Ramon. [user] Yes that sounds good. Do they have anything available at half past 5 in the evening? [system] On what date? [user] Try for 10th of March.',
            slots=OrderedDict([
                ('appointment_date', '10th of March'),
                ('appointment_time', 'half past 5 in the evening'),
                ('city', 'San Ramon'),
                ('dentist_name', 'Perfect Smiles Family Dentistry'),
                ('offers_cosmetic_services',
                 'True of possible values True, False')
            ]))
    ],
    'Services_3': [
        Prompt(
            utt='[user] I need to find a doctor for my earache. [system] Where do you live? [user] I need a GP working in the San Francisco area. [system] I have a general practitioner in San Francisco at the Arthur H Coleman Medical Center: Dickey Jan V MD. [user] That would do. [system] Do you want to make an appointment? [user] Yes please book it for the 8th of March around 15:30.',
            slots=OrderedDict([
                ('city', 'San Francisco'),
                ('type',
                 'General Practitioner of possible values Gynecologist, ENT Specialist, Ophthalmologist, General Practitioner, Dermatologist'
                ),
                ('doctor_name',
                 'Arthur H Coleman Medical Center: Dickey Jan V MD'),
                ('appointment_date', '8th of March'),
                ('appointment_time', '15:30')
            ])),
        Prompt(
            utt="[user] I'm trying to find an ophthalmologist in San Rafael [system] Dr. Kathryn K. Najafi-Tagol, MD in San Rafael comes recommended [user] Great, what's the weather like there on March 1st? [system] 80 degrees with a 2 % chance of rain [user] Ok, please make me an appointment at 11 am.",
            slots=OrderedDict([
                ('appointment_date', 'March 1st'),
                ('appointment_time', '11 am'), ('city', 'San Rafael'),
                ('doctor_name', 'Dr. Kathryn K. Najafi-Tagol, MD'),
                ('type',
                 'Ophthalmologist of possible values Gynecologist, ENT Specialist, Ophthalmologist, General Practitioner, Dermatologist'
                )
            ])),
        Prompt(
            utt='[user] I need a gynecologist in San Jose for a checkup [system] Can I offer Anjali Tate, M.D.? They have good reviews [user] I want their number and address [system] They are at 2400 Samaritan Drive #105 and you can get them on 510-845-8035 [user] Okay, are they open March 8th? [system] I think so, what time is best? [user] 9:45 am',
            slots=OrderedDict([
                ('appointment_date', 'March 8th'),
                ('appointment_time', '9:45 am'), ('city', 'San Jose'),
                ('doctor_name', 'Anjali Tate, M.D.'),
                ('type',
                 'Gynecologist of possible values Gynecologist, ENT Specialist, Ophthalmologist, General Practitioner, Dermatologist'
                )
            ])),
        Prompt(
            utt="[user] Find me an eye specialist in Oakland. [system] I got you, Bay Area Retina Associates is in Oakland. [user] That's good, when am I free on next Tuesday. [system] next Tuesday, you are free from 8 am to 12 pm. [user] Good, book me an appointment for 11:00 am that day.",
            slots=OrderedDict([
                ('appointment_date', 'March 5th'),
                ('appointment_time', '11 am'), ('city', 'Oakland'),
                ('doctor_name', 'Bay Area Retina Associates'),
                ('type',
                 'Ophthalmologist of possible values Gynecologist, ENT Specialist, Ophthalmologist, General Practitioner, Dermatologist'
                )
            ])),
        Prompt(
            utt="[user] I would like to locate a Dermatologist in Campbell. [system] I found 2. How about Age Defying Dermatology? [user] Are there any others? [system] Another would be Amelia K. Hausauer, MD [user] Yes, I'd like to schedule an appointment with them on the 6th of this month at 4:30 pm",
            slots=OrderedDict([
                ('appointment_date', '6th of this month'),
                ('appointment_time', '4:30 pm'), ('city', 'Campbell'),
                ('doctor_name', 'Amelia K. Hausauer, MD'),
                ('type',
                 'Dermatologist of possible values Gynecologist, ENT Specialist, Ophthalmologist, General Practitioner, Dermatologist'
                )
            ]))
    ],
    'Services_4': [
        Prompt(
            utt="[user] Can you find me a psychologist in Pleasant Hill? [system] Sure. I've found David A. Flakoll. He's a psychologist in Pleasant Hill. [user] Yeah I like the sound of him, can you book me an appointment for the 7th at 4 pm please?",
            slots=OrderedDict([
                ('city', 'Pleasant Hill'),
                ('type',
                 'Psychologist of possible values Psychologist, Family Counselor, Psychiatrist'
                ), ('therapist_name', 'David A. Flakoll'),
                ('appointment_date', 'the 7th'), ('appointment_time', '4 pm')
            ])),
        Prompt(
            utt="[user] Find therapists for me, I feel down. [system] City to look in? What kind psychologist, family counselor, etc. [user] Martinez, find a psychiatrist. [system] I found 2. Andreas R. Bollinger is a psychiatrist in Martinez. [user] Good. What's the forecast there for the 10th of this month? [system] Average of 79 degrees and 20 percent chance of rain. [user] Ok. Book that therapist then. [system] Time of appointment? [user] 2:30 in the afternoon.",
            slots=OrderedDict([
                ('appointment_date', '10th of this month'),
                ('appointment_time', '2:30 in the afternoon'),
                ('city', 'Martinez'),
                ('therapist_name', 'Andreas R. Bollinger'),
                ('type',
                 'Psychiatrist of possible values Psychologist, Family Counselor, Psychiatrist'
                )
            ])),
        Prompt(
            utt='[user] Will you find me a therapist? [system] Which city? [user] Redwood City. [system] What kind of therapist? [user] Psychologist. [system] I found a Psychologist, Becker Hubert J in Redwood City. [user] Great. [system] Do you want to schedule an appointment? [user] Yes, please. [system] What date? [user] next Wednesday at 10:30 am.',
            slots=OrderedDict([
                ('appointment_date', 'next Wednesday'),
                ('appointment_time', '10:30 am'), ('city', 'Redwood City'),
                ('therapist_name', 'Becker Hubert J'),
                ('type',
                 'Psychologist of possible values Psychologist, Family Counselor, Psychiatrist'
                )
            ])),
        Prompt(
            utt="[user] I've been having issues with my parents, talking to a family counselor might help. Can you find one in Vallejo? [system] You may wanna try Wetzler Kathryn, I've heard good things. [user] Yes that great. Can you please make an appointment at 4 pm? [system] What date should the appointment for? [user] Can you make it on 8th of this month?",
            slots=OrderedDict([
                ('appointment_date', '8th of this month'),
                ('appointment_time', '4 pm'), ('city', 'Vallejo'),
                ('therapist_name', 'Wetzler Kathryn'),
                ('type',
                 'Family Counselor of possible values Psychologist, Family Counselor, Psychiatrist'
                )
            ])),
        Prompt(
            utt="[user] Find me a Psychologist in the Corte Madera area please [system] There's a Psychologist in Corte Madera that might work for you, Jeff Greenwald [user] OK. Can you schedule me with him? [system] Sure, what time? [user] How about in the evening 5:30 on the 2nd? [system] Confirming your 5:30 pm appointment tomorrow with Jeff Greenwald yes? [user] Actually change that to morning 10 and on the 12th of this month.",
            slots=OrderedDict([
                ('appointment_date', '12th of this month'),
                ('appointment_time', 'morning 10'), ('city', 'Corte Madera'),
                ('therapist_name', 'Jeff Greenwald'),
                ('type',
                 'Psychologist of possible values Psychologist, Family Counselor, Psychiatrist'
                )
            ]))
    ],
    'Trains_1': [
        Prompt(
            utt="[user] I would like to find a train. [system] Where are you departing from and what is your destination? [user] I'm leaving Washington to Chicago on March 6th. [system] There are 5 trains. There is one leaving at 9:40 am for $109. [user] I would like to purchase one ticket. Flexible ticket, and add travel insurance.",
            slots=OrderedDict([
                ('date_of_journey', 'March 6th'), ('from', 'Washington'),
                ('to', 'Chicago'), ('journey_start_time', '9:40 am'),
                ('number_of_adults', '1 of possible values 1, 2, 3, 4, 5'),
                ('trip_protection', 'True of possible values True, False'),
                ('class',
                 'Flexible of possible values Value, Flexible, Business')
            ])),
        Prompt(
            utt='[user] I would like a train, the best value you can find. [system] What date and departure city? [user] The 10th of March from Sacramento to Fresno [system] I have 7, one departing at 8 am for $66. [user] I want to book it. [system] How many and do you want insurance? [user] One seat and yes for insurance.',
            slots=OrderedDict([
                ('class', 'Value of possible values Value, Flexible, Business'),
                ('date_of_journey', '10th of March'), ('from', 'Sacramento'),
                ('journey_start_time', '8 am'),
                ('number_of_adults', '1 of possible values 1, 2, 3, 4, 5'),
                ('to', 'Fresno'),
                ('trip_protection', 'True of possible values True, False')
            ])),
        Prompt(
            utt='[user] I need 1 seat on a train on March 13th. I need to go to Washington from Atlanta. [system] A train is leaving at 1 pm. The value fare would be $115. [user] Perfect, book me that train. [system] Do you want travel insurance? [user] Yes, I want to buy insurance. What station is it leaving from?',
            slots=OrderedDict([
                ('class', 'Value of possible values Value, Flexible, Business'),
                ('date_of_journey', 'March 13th'), ('from', 'Atlanta'),
                ('journey_start_time', '1 pm'),
                ('number_of_adults', '1 of possible values 1, 2, 3, 4, 5'),
                ('to', 'Washington'),
                ('trip_protection', 'True of possible values True, False')
            ])),
        Prompt(
            utt='[user] I need a train going to LA. [system] On what date? [user] on the 8th of this month. [system] Where will you be leaving from? [user] From Sacramento. [system] There is one departing at 6:30 am costing $90 for the value ticket and $150 for business class. [user] I like the sound of that. [system] Will I book you some tickets? [user] Yes that would be good, two business class seats and add trip insurance.',
            slots=OrderedDict([
                ('class',
                 'Business of possible values Value, Flexible, Business'),
                ('date_of_journey', '8th of this month'),
                ('from', 'Sacramento'), ('journey_start_time', '6:30 am'),
                ('number_of_adults', '1 of possible values 1, 2, 3, 4, 5'),
                ('to', 'LA'),
                ('trip_protection', 'True of possible values True, False')
            ])),
        Prompt(
            utt="[user] I need to find 3 flight tickets on March 6th, Washington to NY. [system] Is American Airlines okay? There is a 9:25 pm flight for a total of $84. [user] Sounds okay but can you check a train instead? [system] There is a train departing at 6:40 am for $147. [user] Please book flexible train tickets. I'll also need travel insurance.",
            slots=OrderedDict([
                ('class',
                 'Flexible of possible values Value, Flexible, Business'),
                ('date_of_journey', 'March 6th'), ('from', 'Washington'),
                ('journey_start_time', '6:40 am'),
                ('number_of_adults', '3 of possible values 1, 2, 3, 4, 5'),
                ('to', 'NY'),
                ('trip_protection', 'True of possible values True, False')
            ]))
    ],
    'Travel_1': [
        Prompt(
            utt="[user] I want to find some child-friendly activities like a free museum over in Atlanta. [system] The Children's Museum of Atlanta is a museum your kids would like. [user] What is the phone number? [system] It is 404-659-5437. [user] Cool.",
            slots=OrderedDict([(
                'category',
                'Museum of possible values Place of Worship, Theme Park, Museum, Historical Landmark, Park, Tourist Attraction, Sports Venue, Shopping Area, Performing Arts Venue, Nature Preserve'
            ), ('free_entry', 'True of possible values True, False'),
                               ('good_for_kids',
                                'True of possible values True, False'),
                               ('location', 'Atlanta')])),
        Prompt(
            utt="[user] Can you find me some child-friendly theme park, entry fee or not? [system] Where should I search? [user] Search in SD. [system] Seaworld fits the bill [user] I'll check it out, can you give me their phone number?",
            slots=OrderedDict([(
                'category',
                'Theme Park of possible values Place of Worship, Theme Park, Museum, Historical Landmark, Park, Tourist Attraction, Sports Venue, Shopping Area, Performing Arts Venue, Nature Preserve'
            ), ('free_entry', 'dontcare of possible values True, False'),
                               ('good_for_kids',
                                'True of possible values True, False'),
                               ('location', 'London')])),
        Prompt(
            utt="[user] I'm looking for some wildlife spots in San Francisco. I want to find something that is child-friendly and with free entry. [system] I have 1 for you. What about Aquarium of the Bay, it is a Nature Preserve. [user] What is their address? [system] They're over at Pier 39. [user] Okay that's perfect thanks.",
            slots=OrderedDict([(
                'category',
                'Nature Preserve of possible values Place of Worship, Theme Park, Museum, Historical Landmark, Park, Tourist Attraction, Sports Venue, Shopping Area, Performing Arts Venue, Nature Preserve'
            ), ('free_entry', 'True of possible values True, False'),
                               ('good_for_kids',
                                'True of possible values True, False'),
                               ('location', 'San Francisco')])),
        Prompt(
            utt="[user] Good morning, can you find a nearby place of worship? I'm visiting San Jose [system] Ok, the Cathedral Basilica of St. Joseph seems popular enough. [user] I've heard of it, are kids allowed inside? Is there an entry fee? [system] Kids are fine, entry is free. [user] That will do then.",
            slots=OrderedDict([(
                'category',
                'Place of Worship of possible values Place of Worship, Theme Park, Museum, Historical Landmark, Park, Tourist Attraction, Sports Venue, Shopping Area, Performing Arts Venue, Nature Preserve'
            ), ('free_entry', 'True of possible values True, False'),
                               ('good_for_kids',
                                'True of possible values True, False'),
                               ('location', 'San Jose')])),
        Prompt(
            utt="[user] Can you suggest a strenuous nature trail around Portland? My kids are not with me [system] There are a few. Wildwood trail, maybe? [user] Something simpler please. [system] Try the Marquam trail then [user] Cool, that is doable. Is there an entry fee? [system] Yes, $6 to enter the park [user] That's fine, I'll go there.",
            slots=OrderedDict([(
                'category',
                'Nature Preserve of possible values Place of Worship, Theme Park, Museum, Historical Landmark, Park, Tourist Attraction, Sports Venue, Shopping Area, Performing Arts Venue, Nature Preserve'
            ), ('free_entry', 'False of possible values True, False'),
                               ('good_for_kids',
                                'False of possible values True, False'),
                               ('location', 'Portland')]))
    ],
    'Weather_1': [
        Prompt(
            utt='[user] I want to check the weather for March 7th. [system] What city are you checking the weather for? [user] I want to check in Dixon. [system] The average temperature is 80 degrees with a 1 percent chance of rain. [user] Okay, thanks',
            slots=OrderedDict([('date', 'March 7th'), ('city', 'Dixon')])),
        Prompt(
            utt="[user] What's the weather like in Stinson Beach today? [system] It should be around 80 degrees Fahrenheit with a 16 percent chance of rain. [user] Alright. That's all. Thank you.",
            slots=OrderedDict([('city', 'Stinson Beach'), ('date', 'today')])),
        Prompt(
            utt='[user] I require to do weather check in San Jose on March 10th 2019 [system] The average day temperature have to be 74 degrees Fahrenheit and there is 18 percent chance of rain [user] sure, cheers',
            slots=OrderedDict([('city', 'San Jose'),
                               ('date', 'March 10th 2019')])),
        Prompt(
            utt='[user] Can you check the weather for the 11th of March? [system] In what city? [user] look in Millbrae [system] It is expected to be about 91 degrees with a 25 percent chance of rain. [user] How about Santa Clara same day?',
            slots=OrderedDict([('city', 'Santa Clara'),
                               ('date', '11th of March')])),
        Prompt(
            utt="[user] I'm traveling soon, can you tell me the weather [system] When and where, then? [user] next monday in chicago",
            slots=OrderedDict([('city', 'chicago'), ('date', 'next monday')]))
    ]
}

MW_SEPARATED_ANNOTATION_PROMPTS = {
    'attraction': [
        Prompt(
            utt="[user] can you found me a swimming pool in the south part of town? [system] unfortunately i was unable to find a pool at the south part of town, however we have some north east and at the centre [user] what about a type of park? [system] there are five, including wandlebury country park in the south. [user] wandlebury country park is fine. can i get the entrance fee? [system] actually, they have free entrance. they are located at wandlebury ring, gog magog hills, babraham. their phone number is 01223243830. their postcode is cb223ae. [user] thanks for the info. i will give them a call. [system] you're welcome. can i help you find anything else? [user] no, that's all i need today. thank you! [system] you're welcome, have a nice day!",
            slots=OrderedDict([
                ('attraction-type', 'park'),
                ('attraction-area',
                 'south of possible values centre, east, north, south, west'),
                ('attraction-name', 'wandlebury country park')
            ]))
    ],
    'bus': [
        Prompt(
            utt="[user] can you help me find a train that leaves cambridge after 9:45 pm? thanks. [system] i can help with that. what is the destination and what day would you like to travel? [user] i am going to bishops stortford on wednesday.  i actually need to leave after 21:45 though. [system] i do not have any trains that match your request. [user] that's disappointing. can you recommend a taxi or bus service? [system] i can set you up for bus tr9984, they leave at 23:29 and the price is 10.10 pounds, are you ok with that? [user] okay. can you also tell me about a museum to go to in the centre? [system] of course, there are actually eleven to choose from. [user] can you make recommendation? [system] i like the museum of archaelogy and anthropology.  it is located at the university of cambridge. [user] thank you, can i have the post code for that museum. [system] yes, the postcode is cb23dz, can i assist with anything else? [user] that is all i needed.  thanks! [system] you're welcome. have a great day.",
            slots=OrderedDict([('bus-leaveat', '21:45'),
                               ('bus-destination', 'bishops stortford'),
                               ('bus-day',
                                'wednesday of possible values wednesday'),
                               ('bus-departure', 'cambridge')]))
    ],
    'hospital': [
        Prompt(
            utt="[user] i got injured and i am looking for a hospital nearby. the hospital should have the acute medical assessment unit department [system] addenbrookes hospital has an acute medical assessment unit. the hospital is at hills rd, cambridge, postcode cb20qq. [user] alright. thanks and that's all i need for today. [system] thank you for using our services.",
            slots=OrderedDict([('hospital-department',
                                'acute medical assessment unit')]))
    ],
    'hotel': [
        Prompt(
            utt="[user] i'm looking for some info on a hotel called the cambridge belfry. [system] this is a cheap 4 star hotel in the west.  it has internet and parking.  are you looking to book a stay? [user] yes can i get a booking for 6 people and 2 nights starting from wednesday. [system] the booking was successful.  your reference number is  9gszpi6r.  is there anything else i can help you with? [user] i think i'm all set. thanks! [system] ok i will book that for you and get you a confirmation number",
            slots=OrderedDict([
                ('hotel-name', 'cambridge belfry'),
                ('hotel-type', 'hotel of possible values guesthouse, hotel'),
                ('hotel-people', '6'),
                ('hotel-day',
                 'wednesday of possible values monday, tuesday, wednesday, thursday, friday, saturday, sunday'
                ), ('hotel-stay', '2'),
                ('hotel-area',
                 'west of possible values centre, east, north, south, west'),
                ('hotel-parking', 'yes of possible values free, no, yes'),
                ('hotel-pricerange',
                 'cheap of possible values expensive, cheap, moderate'),
                ('hotel-stars', '4'),
                ('hotel-internet', 'yes of possible values free, no, yes')
            ]))
    ],
    'restaurant': [
        Prompt(
            utt="[user] i am looking for an italian restaurant in the west part of town [system] okay i have found 2 restaurants. one is in the moderate price range and one is in the cheap price range. [user] i want the cheap one please [system] la margherita fits the bill.  great place.  can i make a reservation for you? [user] yes, please. i'd like it for 3 people at 6:45pm on saturday. [system] that booking was successful. your reference number is 3lwl9xaa. [user] thank you for all your help. have a great day. [system] happy to be of help, and i hope you enjoy your meal!",
            slots=OrderedDict([
                ('restaurant-food', 'italian'),
                ('restaurant-area',
                 'west of possible values centre, east, north, south, west'),
                ('restaurant-pricerange',
                 'cheap of possible values cheap, expensive, moderate'),
                ('restaurant-people', '3'),
                ('restaurant-day',
                 'saturday of possible values monday, tuesday, wednesday, thursday, friday, saturday, sunday'
                ), ('restaurant-time', '18:45'),
                ('restaurant-name', 'la margherita')
            ]))
    ],
    'taxi': [
        Prompt(
            utt="[user] can you find me a taxi from backstreet bistro? i can't leave until after 14:15. [system] ok, what is your destination? [user] i want to be taken to cineworld cinema [system] when do you need to arrive? [user] by 15 00 and i need car type and contact number [system] booking completed! booked car type : black skoda contact number : 07336608269 [user] thanks for all your help. goodbye. [system] what time do you need to be at the cineworld cinema?",
            slots=OrderedDict([('taxi-leaveat', '14:15'),
                               ('taxi-departure', 'backstreet bistro'),
                               ('taxi-destination', 'cineworld cinema'),
                               ('taxi-arriveby', '15:00')]))
    ],
    'train': [
        Prompt(
            utt="[user] is there a train from london liverpool street that arrives by 17:15? [system] yes, i have several trains that can get you there by 17:15. what day are you traveling? [user] i'm coming to cambridge on thursday. [system] there are seven trains available that day. what time would you like to leave by? [user] any train is fine. i'll need it booked for three people. [system] i've booked you for 3 on tr3257, leaving liverpool street at 13:39 and arriving cambridge at 15:07, with reference #yr897wtp. can i help you with anything else? [user] yes. i'm also looking for a place called cineworld cinema [system] okay it's located in the south and the address is cambridge leisure park, clifton way. [user] thank you, goodbye! [system] good bye, enjoy the rest of your day.",
            slots=OrderedDict([
                ('train-arriveby', '17:15'),
                ('train-departure', 'london liverpool street'),
                ('train-destination', 'cambridge'),
                ('train-day',
                 'thursday of possible values monday, tuesday, wednesday, thursday, friday, saturday, sunday'
                ), ('train-people', '3'), ('train-leaveat', 'dontcare')
            ]))
    ]
}
