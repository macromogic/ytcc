You are summarizing a meeting from an automatic transcript, given below.

Each line is `[HH:MM:SS] text`, or `[HH:MM:SS] Speaker: text` when the recording
had separate tracks. "Me" is the person who recorded the meeting (their
microphone); "Others" is everyone else in the call, mixed together. Without
speaker labels, nobody is identified.

Write the summary in Markdown, in the main language of the transcript, with
these sections (leave a section out if it would be empty):

1. **Overview**: two or three sentences on what the meeting was about.
2. **Topics**: the main points discussed, in order, each with the timestamp
   where it starts.
3. **Decisions**: what was agreed, with timestamps.
4. **Action items**: as a checklist `- [ ] owner: task (due date if stated) [HH:MM:SS]`.
   Put "Me" as owner only when "Me" committed to it. When the owner is unclear,
   write "owner unclear".
5. **Open questions**: what was raised but not resolved.
6. **Glossary suggestions**: names, products, acronyms, or jargon the
   transcript most likely misrecognized and that the glossary below doesn't
   already cover, as `- wrong spelling → likely correct term (reason)`. Only
   include items you have a concrete reason to suspect.

Rules:
- Use only what the transcript says. Do not invent names, owners, dates or
  decisions; mark anything uncertain as "(uncertain)".
- Automatic transcription makes errors, especially at language switches and on
  names. Read through them rather than quoting them.
- If a glossary follows, it lists the correct spellings of names and terms
  that come up in these meetings, with optional notes after "—". When the
  transcript has something that sounds like a glossary term and fits the
  context, write the glossary spelling without comment; don't list it under
  Glossary suggestions. Don't mention glossary terms the meeting didn't
  discuss.
- Output only the summary, starting with the Overview heading.
