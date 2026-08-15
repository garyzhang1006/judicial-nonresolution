"""Frozen annotation guidelines.

Version 1.0 was drafted before the pilot. Version 1.1 incorporates the single
revision permitted by the study design, made once after the 300-item pilot and
before any main-sample annotation. The revision is recorded in CHANGELOG below
rather than folded silently into the text.
"""

VERSION = "1.1"

CHANGELOG = """
v1.0 -> v1.1 (after pilot, before main sample; one revision, as pre-specified)
 * Added rule L1-6 on partial resolution. The pilot surfaced opinions that
   resolve an issue for one party or one statutory subsection while expressly
   leaving it open for another. v1.0 gave no instruction and annotators split.
   v1.1 fixes the unit of analysis to the proposition named in the anchor span
   and labels partial dispositions UNRESOLVED only when the anchored
   proposition itself is left open.
 * Clarified rule L1-3 to state that a court's characterization of what it
   "need not" do controls only where no independent resolution follows.
 * Added rule L1-9 on issues the opinion never takes up. The pilot surfaced
   passages whose only trigger sits inside a parenthetical or a description of
   what some other court did, on a question this opinion never reaches in its
   own right. v1.0 forced a choice between DECIDED and UNRESOLVED, neither of
   which is true; v1.1 routes these to UNCLEAR. This uses an existing label
   rather than adding one.
No label definitions were added, removed, merged, or renamed.
"""

# --------------------------------------------------------------------------
LAYER1 = """
TASK. You are given a legal issue and a passage from a United States federal
appellate opinion. Decide whether the opinion resolved that issue.

LABELS.
  DECIDED     The authoring court resolved the issue. It stated a holding, a
              conclusion, or a determination on the issue, whether or not that
              resolution was necessary to the judgment, and whether or not it
              was later characterized as dictum.
  UNRESOLVED  The authoring court did not resolve the issue for itself.

If and only if the label is UNRESOLVED, assign one sublabel.
  AVOIDED     The court declined to reach the issue because some other ground
              disposed of the case or the claim. Typical form: "because we
              affirm on qualified immunity grounds, we need not decide whether
              the search was lawful."
  ASSUMED     The court proceeded on a stated assumption about the issue and
              resolved the case on other grounds, so that the assumption did
              no independent work. Typical form: "assuming without deciding
              that the statute reaches this conduct, the evidence is
              insufficient."
  RESERVED    The court expressly held the issue open for future resolution,
              signalling that it remains available. Typical form: "we leave for
              another day whether Chevron survives in this context."

DECISION RULES.
  L1-1  ATTRIBUTION. The label describes what THIS opinion did. Language about
        what a different court did, whether a lower court, a sister circuit, or
        the Supreme Court, does not by itself make this opinion's disposition
        UNRESOLVED. "The district court concluded it need not decide X" tells
        you about the district court. Read on to see what this court did.
  L1-2  UNIT. In a cluster containing a majority opinion and separate opinions,
        each opinion is its own unit. A dissent that would decide an issue the
        majority avoided is DECIDED for the dissent and UNRESOLVED for the
        majority.
  L1-3  LATER RESOLUTION CONTROLS. A court may write "we need not decide X" and
        then decide X anyway, in the same opinion, in an alternative holding, in
        a footnote, or in a later section. Where the opinion does resolve the
        issue somewhere, the label is DECIDED. A court's characterization of
        what it need not do controls only where no independent resolution
        follows.
  L1-4  ASSUMPTION THAT DOES WORK. If the court assumes a proposition and the
        assumption is then treated as settled for the remainder of the analysis
        in a way that determines the outcome, the label remains UNRESOLVED and
        the sublabel is ASSUMED. Assuming a proposition is not deciding it, even
        when the assumption drives the result.
  L1-5  NO TRIGGER REQUIRED. UNRESOLVED does not require any particular phrase.
        A court can leave an issue open in ordinary prose.
  L1-6  PARTIAL RESOLUTION. The unit is the proposition named in the issue
        statement, not the broader legal question it belongs to. If the court
        resolves the anchored proposition but leaves a neighbouring one open,
        the label is DECIDED. If the court resolves a neighbouring proposition
        but leaves the anchored one open, the label is UNRESOLVED.
  L1-7  REMANDS. An issue remanded for the district court to address in the
        first instance is UNRESOLVED, sublabel AVOIDED, unless the appellate
        court also states the governing rule and applies it.
  L1-8  ABSTAIN. If the passage is too corrupt, too truncated, or too ambiguous
        to support either label, output UNCLEAR rather than guessing.
  L1-9  ISSUES THIS OPINION NEVER TAKES UP. Sometimes the only reason an issue
        is in front of you is that the opinion mentioned, in a parenthetical or
        in a description of some other court's work, that a different court did
        not decide it, and this opinion never reaches the issue in its own
        right. That is neither DECIDED nor UNRESOLVED for this opinion. Output
        UNCLEAR. This rule does not apply where the opinion goes on to engage
        with the issue itself: there, L1-1 and L1-3 govern.
"""

# --------------------------------------------------------------------------
LAYER2 = """
TASK. You are given (a) an issue that an earlier federal appellate opinion left
unresolved or decided, together with the passage from that earlier opinion, and
(b) a passage from a later opinion that cites the earlier one. Characterize how
the later opinion treats the earlier one on that issue.

STATUS. What decisional force does the later opinion ascribe to the earlier
opinion on this issue?
  L0  Explicitly unresolved. The later opinion states or plainly conveys that
      the earlier court did not decide the issue.
  L1  Assumed or conditional. The later opinion treats the earlier opinion as
      having assumed, suggested, or proceeded on the issue without deciding it.
  L2  Supported or probable. The later opinion cites the earlier opinion as
      support for the proposition without saying it was held, for example with
      "see", "cf.", or "see generally", or by describing the earlier court as
      having indicated or suggested it.
  L3  Established. The later opinion states or plainly implies that the earlier
      opinion held, established, or decided the proposition, or cites it as
      controlling authority for the proposition without qualification.

ATTRIBUTION A. Set A=1 if the later opinion attributes a resolution of the issue
to the earlier opinion. Otherwise A=0. A=1 requires that the earlier opinion be
named or cited as the source of the proposition.

INDEPENDENT RESOLUTION E. Set E=1 if the later opinion itself resolves the issue
on its own analysis, whatever it says about the earlier opinion. Otherwise E=0.

DECISION RULES.
  L2-1  A and E are independent. A later court may correctly note that the issue
        was left open and then resolve it itself: A=0, E=1. This is ordinary
        doctrinal development, not misattribution.
  L2-2  Escalation is a comparison of decisional status, not a judgment of
        correctness or good faith. Do not label based on whether the later
        court reached the right answer.
  L2-3  The citing passage must be about the same issue. If the later opinion
        cites the earlier one for something else, output OFF_ISSUE.
  L2-4  Where the later opinion cites the earlier one only in a string citation
        with no accompanying characterization, use L2 if the string supports the
        proposition and OFF_ISSUE if it does not.
  L2-5  Quoting the earlier court's own non-resolution language is L0, even if
        the later court then relies on the surrounding reasoning.
  L2-6  If the passage is too corrupt or ambiguous to characterize, output
        UNCLEAR.
"""

# --------------------------------------------------------------------------
ISSUE_STATEMENT = """
TASK. Write the legal issue that the marked passage is about, as a single
neutral clause beginning with "whether".

CONSTRAINTS.
  I-1  Neutral. The statement must not reveal or hint at how, or whether, any
       court came out. Do not use "held", "decided", "declined", "assumed",
       "reserved", "left open", "need not", or "without deciding".
  I-2  Self-contained. A reader who has not seen the opinion must be able to
       understand what is at stake. Name the statute, doctrine, or standard.
  I-3  Faithful. State the issue the passage is actually about, at the level of
       generality the passage uses.
  I-4  One clause, at most forty words.
  I-5  If the marked passage does not identify a legal issue at all, output
       NO_ISSUE.

EXAMPLES.
  Passage: "Because we resolve the case on standing grounds, we need not decide
  whether the statute violates the First Amendment."
  Issue: whether the statute violates the First Amendment

  Passage: "We hold that the district court correctly applied the categorical
  approach to Mr. Reyes's prior conviction."
  Issue: whether the categorical approach governs the classification of the
  defendant's prior conviction
"""
