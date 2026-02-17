def identify_roles(aligned_data):

    stats = {}

    for item in aligned_data:
        spk = item["speaker"]
        duration = item["end"] - item["start"]

        if spk not in stats:
            stats[spk] = {
                "duration": 0,
                "questions": 0
            }

        stats[spk]["duration"] += duration

        if "？" in item["text"] or "吗" in item["text"]:
            stats[spk]["questions"] += 1

    speakers = list(stats.keys())

    if stats[speakers[0]]["questions"] > stats[speakers[1]]["questions"]:
        therapist = speakers[0]
    else:
        therapist = speakers[1]

    roles = {}

    for spk in speakers:
        if spk == therapist:
            roles[spk] = "咨询师"
        else:
            roles[spk] = "来访者"

    return roles
