from app.rag.qa_pipeline import (
    extract_section_answer,
    generate_rag_result,
    is_contents_like_text,
    section_heading_score,
)


def test_contents_page_is_not_treated_as_section_body():
    text = (
        "CONTENTS CHAPTER XVII Of Offences Against Property "
        "420. Cheating and dishonestly inducing delivery of property. "
        "Of Fraudulent Deeds and Dispositions of Property "
        "421. Dishonest or fraudulent removal or concealment of property "
        "to prevent distribution among creditors. "
        "422. Dishonestly or fraudulently preventing debt being available for creditors."
    )

    assert is_contents_like_text(text)
    assert section_heading_score("420", text) == 0
    assert extract_section_answer("420", text) == ""


def test_section_420_body_stops_before_section_421():
    text = (
        "420. Cheating and dishonestly inducing delivery of property. "
        "Whoever cheats and thereby dishonestly induces the person deceived "
        "to deliver any property to any person, or to make, alter or destroy "
        "the whole or any part of a valuable security, shall be punished with "
        "imprisonment of either description for a term which may extend to seven "
        "years, and shall also be liable to fine. "
        "421. Dishonest or fraudulent removal or concealment of property to "
        "prevent distribution among creditors. Whoever dishonestly or fraudulently "
        "removes property shall be punished."
    )

    answer = extract_section_answer("420", text)

    assert section_heading_score("420", text) >= 100
    assert answer.startswith("420. Cheating and dishonestly inducing delivery of property.")
    assert "Whoever cheats" in answer
    assert "421. Dishonest" not in answer


def test_section_title_starting_with_act_is_valid():
    text = (
        "508. Act caused by inducing person to believe that he will be rendered "
        "an object of the Divine displeasure. Whoever voluntarily causes or attempts "
        "to cause any person to do anything which that person is not legally bound "
        "to do, by inducing or attempting to induce that person to believe that he "
        "will become by some act of the offender an object of Divine displeasure, "
        "shall be punished. 509. Word, gesture or act intended to insult the "
        "modesty of a woman."
    )

    answer = extract_section_answer("508", text)

    assert section_heading_score("508", text) >= 100
    assert answer.startswith("508. Act caused")
    assert "shall be punished" in answer
    assert "509. Word" not in answer


def test_section_508_contents_cluster_is_rejected():
    text = (
        "508. Act caused by inducing person to believe that he will be rendered "
        "an object of the Divine displeasure. 509. Word, gesture or act intended "
        "to insult the modesty of a woman. 510. Misconduct in public by a drunken "
        "person. CHAPTER XXIII OF ATTEMPTS TO COMMIT OFFENCES 511. Punishment for "
        "attempting to commit offences punishable with imprisonment for life or other."
    )

    assert is_contents_like_text(text)
    assert section_heading_score("508", text) == 0
    assert extract_section_answer("508", text) == ""


def test_generate_section_result_uses_body_not_semantic_rag():
    chunks = [
        {
            "page": 13,
            "chunk_index": 1,
            "text": (
                "508. Act caused by inducing person to believe that he will be rendered "
                "an object of the Divine displeasure. 509. Word, gesture or act intended "
                "to insult the modesty of a woman. 510. Misconduct in public by a drunken "
                "person. CHAPTER XXIII OF ATTEMPTS TO COMMIT OFFENCES 511. Punishment for "
                "attempting to commit offences punishable with imprisonment for life or other."
            ),
        },
        {
            "page": 96,
            "chunk_index": 1,
            "text": (
                "508. Act caused by inducing person to believe that he will be rendered "
                "an object of the Divine displeasure. Whoever voluntarily causes or attempts "
                "to cause any person to do anything which that person is not legally bound "
                "to do, by inducing or attempting to induce that person to believe that he "
                "will become by some act of the offender an object of Divine displeasure, "
                "shall be punished. 509. Word, gesture or act intended to insult the "
                "modesty of a woman."
            ),
        },
    ]

    result = generate_rag_result("section 508", chunks, index=None)

    assert result["confidence"] == "high"
    assert result["pages"] == [96]
    assert "Whoever voluntarily causes" in result["answer"]
    assert "509. Word" not in result["answer"]


def test_missing_section_returns_no_confidence():
    chunks = [
        {
            "page": 1,
            "chunk_index": 1,
            "text": "420. Cheating and dishonestly inducing delivery of property. Whoever cheats shall be punished.",
        }
    ]

    result = generate_rag_result("section 999", chunks, index=None)

    assert result["confidence"] == "none"
    assert result["sources"] == []
    assert "Section 999" in result["answer"]


if __name__ == "__main__":
    test_contents_page_is_not_treated_as_section_body()
    test_section_420_body_stops_before_section_421()
    test_section_title_starting_with_act_is_valid()
    test_section_508_contents_cluster_is_rejected()
    test_generate_section_result_uses_body_not_semantic_rag()
    test_missing_section_returns_no_confidence()
    print("section retrieval regression tests passed")
