import soundDownload as SD

queries = ["bassoon", "cello", "clarinet", "daluo", "flute_note", "mridangam", "naobo", "snare", "trumpet", "violin"]
tags = [None, "single-note", "single-note", "icassp2014-dataset", "good-sounds", "icassp2013-dataset", "icassp2014-dataset", "one-shot", "good-sounds", "single-note"]
duration = (0,5)
key = "hjNYO8PPFWKuvcboJo9YEhtGRSv8TFwNsYY5LWVM"
output = "sounds"

for i in range(len(queries)):
    print("Downloading sounds for \"{}\" query, with \"{}\" tag...".format(queries[i], tags[i]))
    if queries[i] == "naobo":
        SD.downloadSoundsFreesound(queryText = queries[i], tag = tags[i], duration = duration, API_Key = key, outputDir = output, topNResults = 20, featureExt = ".json")
    print("... Done")

