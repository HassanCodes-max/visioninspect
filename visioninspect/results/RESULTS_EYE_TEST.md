# Early look: can a vision LLM judge one photo?

Run on 2026-11-02 by Claude Opus 5.5 inside the working session, looking at each photo directly. This was not an API test. It is a lead for choosing the next test, and it is not a measured claim for the site.

## Why it was run

A visitor uploaded two leather photos from the web. Both went to REVIEW, even though one had an obvious tear and "Leather" is on the site's list. Recognition closeness was 0.55 and 0.51, against a cut-off of 0.85. The defect model scored both near 1.0, including the clean one, so removing the recognition step would not have helped.

## Method

32 VisA photos: 4 product types, 4 good and 4 defective of each, chosen at random and given random names. Each photo got one GOOD or DEFECT answer with no reference photo. All answers were saved before the key was opened.

## Result

| Measure | Result |
|---|---:|
| Correct | 31 of 32 |
| Defects caught | 16 of 16 |
| Good items passed | 15 of 16 |
| Balanced accuracy | 96.9% |

The one error was a good candle photo called defective because of a thin line on its surface.

## Limits

- 32 photos is small. With 31 of 32 correct, the true rate could be roughly 84% to 99%.
- Candles and chewing gum had been seen earlier in this session. Pipe fryums and circuit boards appeared only in the blind set, but all four are from one dataset of studio photos.
- A fixed prompt through the API may behave differently, and cheaper models are untested.
- Photos from the web or a phone have not been tested in this way.

## Next

A proper test needs a fixed prompt and rules set in advance, at least several hundred photos from products beyond these four, and phone photos. It needs an API key and costs money per photo.
