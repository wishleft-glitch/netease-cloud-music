# PMEmo review-only emotion tagging pilot

This project generates review-only PMEmo emotion-tag candidates. It does not emit final labels: every candidate has `is_final_label=False` and requires human review.

Run the pilot from PowerShell:

```powershell
Set-Location 'F:\netease\_music\emotion_tagging'
$env:PYTHONPATH = "$PWD\src"
& "C:\Users\mss\AppData\Local\Programs\Python\Python312\python.exe" -m emotion_tagging.cli --pmemo-root "F:\netease\_music\PMEmo\data_2019\extracted\PMEmo2019" --tag-cards ".\config\pilot_tag_cards.json" --output-dir ".\outputs"
```

The generated CSV and JSON are candidate evidence for review. Official emotion definitions and gold data are required before calibration and any final-label use.

## Pilot result

- Candidate rows: 478
- Candidates by tag: 活力 351, 思念 79, 热血 33, 治愈 10, 孤独 5
- Lyric coverage: 767 songs total; 606 with lyrics; 161 missing lyrics
- Duplicate `music_id`/tag pairs: none

PMEmo candidates are not final labels and must not be used to claim the competition's 95% precision target. Official tag definitions, authorized competition songs and an independent human-labelled evaluation set are required before production calibration.
