"""Tests for the Hinglish (Romanized Hindi) coverage in the engine's word
lists: question / command / connector / pronoun detection, topic words, and
the dossier's stop-words. The benefit of these additions on real Hinglish chats
is unmeasured (the demo chats are English-heavy); these tests pin what they do.
Run: pytest test_hinglish_language.py
"""
from detective import HINGLISH_CATEGORY_WORDS, category_features, has_category_signal, syntactic_features
from person_profile import STOPWORDS, content_words


def test_hinglish_question_without_a_question_mark():
    assert syntactic_features("kya kar raha hai")["question"] == 1.0
    assert syntactic_features("kab aa raha hai tu")["question"] == 1.0
    assert syntactic_features("bhej de yaar")["question"] == 0.0


def test_hinglish_command_is_an_imperative():
    assert syntactic_features("bhej de yaar jaldi")["imperative"] == 1.0
    assert syntactic_features("dekh zara")["imperative"] == 1.0


def test_hinglish_connectors_count_towards_clause_density():
    assert syntactic_features("main aayunga lekin late hoga")["clause_density"] > 0
    assert syntactic_features("kal milte hain")["clause_density"] == 0


def test_hinglish_pronouns_feed_the_self_vs_other_ratio():
    # main (I) + tujhe (you) -> balanced; only "main" -> all self
    assert syntactic_features("main tujhe bataunga")["person_ratio"] == 0.5
    assert syntactic_features("main aa raha hoon")["person_ratio"] == 1.0


def test_hinglish_topic_words_are_recognised():
    assert has_category_signal("aaj chai peene chalein")
    assert category_features("kal padhai karni hai")["studies"] > 0.5
    assert category_features("khana kha liya kya")["food"] > 0.5
    assert category_features("bhai kaam bahut hai")["work"] > 0.5
    # a message with no topic word still reads as "no signal", not as a topic
    assert not has_category_signal("haan theek hai bhai")


def test_english_words_are_untouched_by_the_additions():
    assert category_features("the exam is tomorrow")["studies"] == 1.0
    assert syntactic_features("what are you doing")["question"] == 1.0


def test_hinglish_function_words_do_not_show_up_as_someones_vocabulary():
    assert content_words("hai tha toh nahi kya main tum") == []
    # ...but the words that carry meaning or personality stay
    assert "biryani" in content_words("aaj biryani banayi")
    assert "bro" in content_words("bro seriously")
    assert "yaar" in content_words("arre yaar")


def test_additions_never_collide_with_common_english_words():
    english_that_must_stay_english = {"do", "par", "to", "so", "the", "in", "on", "me", "is", "at", "let", "will"}
    every_addition = set().union(*HINGLISH_CATEGORY_WORDS.values())
    assert not (every_addition & english_that_must_stay_english)
    assert not (STOPWORDS & {"bro", "bhai", "yaar", "arre"})
