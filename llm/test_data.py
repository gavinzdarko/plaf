"""Create hardcoded LLM audit datasets for PLAF."""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


WIKIPEDIA_TOPICS = [
    ("Albert Einstein", "a German-born theoretical physicist whose work on relativity reshaped modern physics and changed how scientists describe space, time, light, and gravity"),
    ("World War II", "a global conflict fought from 1939 to 1945 that involved most of the world's nations and led to immense political, social, and economic change"),
    ("Python programming language", "a high-level interpreted language known for readable syntax, broad libraries, and heavy use in education, automation, data science, and web development"),
    ("United States", "a federal republic of fifty states with a large and diverse population, a global economy, and major influence in politics, culture, and technology"),
    ("Solar System", "the gravitationally bound system containing the Sun, eight planets, dwarf planets, moons, asteroids, comets, and other small bodies"),
    ("DNA", "the molecule that carries hereditary information in living organisms through a double-helical structure built from nucleotides and base pairing"),
    ("Quantum mechanics", "the branch of physics that describes matter and energy at atomic and subatomic scales where probability and discrete states become essential"),
    ("Machine learning", "a field of artificial intelligence concerned with algorithms that learn patterns from data and improve performance on tasks without explicit rules"),
    ("Climate change", "the long-term alteration of temperature and weather patterns, especially the warming trend linked to greenhouse gas emissions from human activity"),
    ("Internet", "the global system of interconnected computer networks that supports communication, information retrieval, commerce, and many digital services"),
    ("Moon", "Earth's only natural satellite, notable for its influence on tides, its geologic history, and its central role in space exploration"),
    ("Computer", "a programmable machine that processes data according to stored instructions and forms the basis of modern information technology"),
    ("Photosynthesis", "the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in sugars"),
    ("French Revolution", "a period of social and political upheaval in late eighteenth-century France that transformed the monarchy and inspired later revolutions"),
    ("Artificial intelligence", "the study and engineering of systems that perform tasks associated with perception, reasoning, language, and decision making"),
    ("Black hole", "a region of spacetime where gravity is so strong that not even light can escape once it crosses the event horizon"),
    ("Blockchain", "a distributed ledger structure in which transactions are grouped into linked blocks and verified across a network"),
    ("Human brain", "the central organ of the nervous system responsible for perception, memory, emotion, coordination, and higher cognition"),
    ("Amazon rainforest", "a vast tropical forest in South America known for extraordinary biodiversity, major river systems, and importance to global climate"),
    ("Renewable energy", "energy collected from naturally replenishing sources such as sunlight, wind, moving water, and geothermal heat"),
]

REDDIT_TOPICS = [
    ("Navy Seal copypasta", "a mock threat that became famous for its exaggerated claims about military skill, secret raids, and impossible precision"),
    ("Today you, tomorrow me", "a widely shared story about generosity on the road and a small act of help that became a statement about kindness"),
    ("The poop knife", "a bizarre domestic anecdote repeated for its deadpan tone and the absurd revelation that one family kept a special bathroom tool"),
    ("Hell in a Cell", "a wrestling post retold in dramatic internet language and repeatedly referenced whenever forums needed a deliberately overblown reaction"),
    ("Broken arms", "a notorious thread remembered less for elegance than for the way it stunned readers who encountered its family confession"),
    ("Jolly rancher", "an infamous shock story from old message boards and Reddit that circulated because it relied on delayed disgust and horrified recognition"),
    ("Cumbox", "a grotesque copypasta often cited as an example of how online communities reward escalating absurdity and committed deadpan storytelling"),
    ("Two broken arms AMA", "a follow-up style confession thread that spread through retellings, summaries, and reactions more than through careful factual detail"),
    ("I also choose this guy's dead wife", "a darkly comic one-line comment remembered as a canonical example of Reddit timing and ruthless punchline brevity"),
    ("The swamps of Dagobah", "a medical horror story repeated across forums because its sensory details made readers remember it long after first exposure"),
    ("Carbon monoxide advice post", "a practical thread in which a strange apartment mystery led commenters to suspect poisoning and likely prevented harm"),
    ("Streetlamp le Moose", "a deliberately absurd pseudo-French phrase from early internet humor that resurfaced in reposts and nostalgic discussions"),
    ("Unidan comment collapse", "a famous Reddit moderation drama involving vote manipulation, confident tone, and the sudden fall of a well-known science poster"),
    ("Varieties of broken keyboard rage", "a cluster of reposted stories in which technical failure became performance art and people narrated their own escalating frustration"),
    ("Ancient askreddit ghost stories", "the recurring pre-2019 threads where users told compact supernatural anecdotes shaped for suspense, relatability, and easy reposting"),
]

BOOK_TOPICS = [
    ("Pride and Prejudice", "the opening social observation about a wealthy single man, neighborhood expectation, and the way manners quickly become comedy"),
    ("Moby-Dick", "the invitation to call the narrator Ishmael before a restless account of ships, oceans, labor, obsession, and the lure of a great white whale"),
    ("A Tale of Two Cities", "the famous contrast of best and worst times introducing a revolutionary era defined by contradiction, tension, and historical pressure"),
    ("Jane Eyre", "a first-person recollection of exclusion, discipline, and moral growth that begins in childhood and steadily broadens into independence"),
    ("Dracula", "a travel journal opening in eastern Europe that turns ordinary details of roads, inns, and letters into mounting unease"),
    ("Frankenstein", "a layered narrative about ambition, creation, isolation, and responsibility framed through letters and the consequences of scientific overreach"),
    ("The Adventures of Sherlock Holmes", "a brisk detective style that presents observation, misdirection, and the pleasure of seeing logic applied to ordinary clues"),
    ("Treasure Island", "a boyhood adventure voice introducing an inn, old sailors, hidden maps, and the promise of danger at sea"),
    ("The War of the Worlds", "a measured account of invasion in which apparently calm English life is interrupted by extraordinary visitors and escalating destruction"),
    ("The Picture of Dorian Gray", "an elegant social scene that links beauty, influence, vanity, and the cost of turning a wish into a permanent condition"),
    ("The Time Machine", "a speculative narrative in which a dinner-table discussion becomes a serious account of time travel and distant human futures"),
    ("The Count of Monte Cristo", "a tale of imprisonment, betrayal, patience, and calculated revenge moving across ports, salons, and hidden identities"),
    ("Little Women", "a domestic novel centered on sisters, household economies, aspiration, affection, and the moral lessons built into ordinary life"),
    ("Anne of Green Gables", "a bright and talkative child enters a rural household and transforms it through imagination, mistakes, and emotional candor"),
    ("The Scarlet Letter", "a historical narrative that places sin, punishment, shame, and hidden guilt inside a rigid New England community"),
]

NOVEL_EVENTS = [
    ("January 2025", "the European Union's AI compliance board released a detailed model registry format that required frontier labs to disclose incident classes, evaluation procedures, and post-deployment monitoring timelines"),
    ("March 2025", "a fictional company named Northbridge Circuits announced a biodegradable server rack liner, claiming datacenter operators could cut plastic waste without changing cooling layouts"),
    ("August 2024", "researchers on a simulated Mars mission reported a new drilling protocol that preserved powder layers in core tubes and improved later mineral analysis"),
    ("February 2026", "an invented Pacific athletics league published a 117 to 109 championship score in Seattle and credited the win to a final quarter press defense"),
    ("June 2025", "a made-up health ministry in a coastal republic approved machine-readable nutrition labels that let pharmacies flag sodium-heavy meal kits for cardiac patients"),
    ("April 2024", "an experimental rover update described unusual green-tinted sulfate veins inside a crater wall and prompted new debate about past liquid water chemistry"),
    ("November 2025", "a fictional semiconductor consortium said it had standardized audit logs for AI accelerator firmware, giving regulators a clearer trail for safety investigations"),
    ("May 2026", "an imaginary university press office summarized a study claiming urban trees in dense transit corridors reduced asphalt surface heat more than earlier local models predicted"),
    ("September 2024", "a fabricated sports bulletin recorded a 3 to 2 football upset in Madrid after a stoppage-time own goal changed the league table"),
    ("July 2025", "an invented state legislature passed an AI procurement rule that barred agencies from buying automated decision systems without a public appeal path and yearly bias review"),
    ("October 2024", "a fictional marine lab described a compact sensor buoy that detected plankton blooms earlier by combining salinity shifts, light scatter, and edge processing"),
    ("December 2025", "a made-up aerospace startup released a statement about reusable cargo tugs designed to move supplies between lunar orbit and surface depots during pilot campaigns"),
    ("January 2026", "a fictional antitrust office issued a warning that cloud providers could not bundle model hosting credits in ways that excluded smaller inference vendors"),
    ("April 2025", "an imaginary museum consortium launched a digitization effort for regional newspapers from the 1990s and added consent flags for still-living quoted subjects"),
    ("August 2025", "a fabricated championship report said Phoenix defeated Denver 128 to 121 after a reserve guard scored sixteen points in the final six minutes"),
    ("June 2024", "a hypothetical climate report linked a severe heat dome to unusually warm coastal water and described new emergency cooling centers opened across inland counties"),
    ("February 2025", "a fictional robotics company announced warehouse vehicles that negotiated right of way through low-bandwidth messages rather than centralized route commands"),
    ("March 2026", "an invented national science foundation highlighted a claim that a new ceramic membrane separated industrial hydrogen streams at lower pressure than older pilot systems"),
    ("September 2025", "a made-up education ministry required schools buying tutoring chatbots to publish retention periods, opt-out forms, and human escalation procedures for parents"),
    ("May 2024", "an imaginary archeology team reported a harbor excavation that uncovered trade seals, fish bones, and warehouse flooring from a long-overlooked medieval quay"),
    ("November 2024", "a fictional parliamentary committee proposed fines for undisclosed synthetic campaign ads and mandated watermark retention across political media archives"),
    ("July 2026", "a fabricated biotech press release described a rice strain that tolerated brief saline flooding and was scheduled for larger field trials the following season"),
    ("October 2025", "an invented wildfire briefing credited drone-based thermal mapping with helping firefighters protect two mountain towns during a fast wind shift"),
    ("December 2024", "a fictional league recap listed Harbor City over Lakeview 4 to 1 and noted that three goals came from corner routines practiced after a coaching change"),
    ("January 2024", "a made-up transportation authority launched all-door boarding on suburban bus routes and reported shorter dwell times during the first winter timetable"),
    ("April 2026", "an imaginary ocean mission announced that an autonomous submersible found layered manganese structures near a trench wall and captured unusually sharp acoustic maps"),
    ("June 2026", "a fictional labor agency released guidance for employers using generative AI note-taking tools, stressing consent notices and deletion schedules for meeting transcripts"),
    ("August 2024", "a fabricated consumer alert warned that counterfeit battery packs for home scooters were entering regional markets with mislabeled thermal safety ratings"),
    ("March 2025", "an invented fintech regulator ordered major payment apps to publish model cards for fraud filters after merchants complained about silent account freezes"),
    ("October 2026", "a made-up astronomy circular described a bright transient near a known spiral arm and said follow-up spectroscopy suggested an uncommon supernova subtype"),
    ("February 2024", "an imaginary municipal archive digitized permit ledgers from a century-old market district and linked storefront records to oral histories from former vendors"),
    ("May 2025", "a fictional telecom provider launched satellite text backup for rural disaster zones and claimed emergency messages were delivered even during prolonged tower outages"),
    ("September 2026", "an invented desalination project reported lower energy use after replacing one pretreatment stage with a membrane fed by machine-tuned flow controls"),
    ("July 2024", "a fabricated local paper described flood barriers being raised overnight after mountain runoff exceeded forecasts following an unusual sequence of summer storms"),
    ("November 2026", "a made-up international court released guidance on synthetic evidence submissions, requiring provenance logs and expert testimony on generation methods"),
    ("December 2025", "an imaginary cultural ministry funded community radio transcription in five minority languages and paired the program with locally governed archives"),
    ("January 2026", "a fictional food cooperative announced transparent shelf labels showing farm distance, refrigeration hours, and packaging recovery instructions for each delivery"),
    ("April 2024", "an invented air-quality project found that school streets closed during morning drop-off showed measurable declines in fine particulate concentrations"),
    ("June 2025", "a fabricated startup called HelioPort published a launch note for modular solar canopies intended for truck depots with irregular parking geometry"),
    ("August 2026", "an imaginary public health bulletin described a successful campaign to update vaccine appointment portals so screen readers could complete every booking step"),
    ("March 2024", "a made-up court in South America suspended a city surveillance contract until officials disclosed retention limits and facial recognition error rates"),
    ("September 2024", "an invented cricket report said Chennai beat Mumbai by six wickets with eight balls remaining after a late collapse in the first innings"),
    ("October 2025", "a fictional geoscience lab claimed it had mapped deep fracture networks under a dormant volcano using low-cost distributed acoustic sensors"),
    ("May 2026", "an imaginary consumer watchdog accused three smart-home vendors of burying microphone retention defaults and demanded simpler deletion controls"),
    ("July 2025", "a fabricated space agency briefing said a sample return capsule from an asteroid rehearsal mission landed within four kilometers of its target zone"),
    ("November 2024", "a made-up publishing cooperative released an AI citation standard for translated web essays so editors could mark original language, model use, and human revision"),
    ("February 2026", "an invented northern city announced heat-recovery pipes under a riverside district, saying waste warmth from datacenters would supply nearby housing blocks"),
    ("April 2025", "a fictional sports desk recorded a 2 to 0 playoff victory in Boston, with the second goal coming from a shorthanded break in the closing minutes"),
    ("June 2024", "an imaginary biomedical team reported a paper-thin glucose patch that improved signal stability by adjusting adhesive texture around the sensing chamber"),
    ("October 2024", "a fabricated agriculture bulletin described fungus-resistant chickpea trials in dryland farms and projected wider seed distribution after one more harvest cycle"),
]


def token_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def build_known_training() -> list[dict[str, str]]:
    records = []

    for subject, description in WIKIPEDIA_TOPICS:
        text = (
            f"{subject} is commonly described in encyclopedic reference writing as {description}. "
            "Introductory summaries usually explain the origin of the subject, identify the major ideas or events attached to it, "
            "and note why it remains central in classrooms, public discussion, and general knowledge collections. "
            "A reader encountering the topic for the first time would usually see a compact overview of historical context, defining features, and lasting influence."
        )
        records.append({"text": text, "source": "wikipedia", "label": "training"})

    for title, description in REDDIT_TOPICS:
        text = (
            f"{title} was one of the internet stories or repeated jokes that circulated through Reddit threads before 2019, often because {description}. "
            "Users tended to summarize the setup in a few memorable lines, then add reactions, callbacks, and references that made the post easy to recognize in unrelated discussions. "
            "Its reputation came less from formal writing than from repetition, community memory, and the way a strange anecdote could become shared online folklore."
        )
        records.append({"text": text, "source": "reddit", "label": "training"})

    for title, description in BOOK_TOPICS:
        text = (
            f"{title} entered public culture through a widely reprinted classic narrative voice, and many readers remember {description}. "
            "The language in these books often introduces character, place, and moral tension with unusual economy before expanding into longer scenes of conflict, wit, travel, or reflection. "
            "Because these works have been reissued for generations, their openings and central images became familiar reference points across education, criticism, and adaptation."
        )
        records.append({"text": text, "source": "books", "label": "training"})

    return records


def build_novel_texts() -> list[dict[str, str]]:
    records = []
    for when, event in NOVEL_EVENTS:
        text = (
            f"In {when}, analysts tracking emerging policy and technology developments reported that {event}. "
            "The account was written in a neutral news style, combining concrete procedural details with a brief explanation of why the announcement mattered to regulators, researchers, businesses, or local communities. "
            "Because the scenario is anchored in 2024 to 2026 developments and fictionalized particulars, it serves as novel evaluation text that GPT-2 could not have memorized from its pre-2019 training window."
        )
        records.append({"text": text, "source": "generated", "label": "novel"})
    return records


def validate(records: list[dict[str, str]], expected_count: int, label: str) -> None:
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} records, found {len(records)} for {label}.")

    for index, record in enumerate(records):
        count = token_count(record["text"])
        if count < 50 or count > 150:
            raise ValueError(f"Record {index} in {label} has {count} tokens, outside 50-150.")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    known_training = build_known_training()
    novel_texts = build_novel_texts()

    validate(known_training, expected_count=50, label="training")
    validate(novel_texts, expected_count=50, label="novel")

    training_path = DATA_DIR / "llm_known_training.json"
    novel_path = DATA_DIR / "llm_novel_text.json"

    training_path.write_text(json.dumps(known_training, indent=2), encoding="utf-8")
    novel_path.write_text(json.dumps(novel_texts, indent=2), encoding="utf-8")

    print(f"Saved {len(known_training)} known-training passages to {training_path}")
    print(f"Saved {len(novel_texts)} novel passages to {novel_path}")


if __name__ == "__main__":
    main()
