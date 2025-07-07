from flask import Blueprint, render_template, request
from concurrent.futures import ThreadPoolExecutor
from functions import *


network = Blueprint(__name__, "network")


def pw(word):
    vowel = vowel_extractor(word)
    harmony = syllable_harmony(word)
    word_info = word_information(word)
    etymology = etymology_analysis(word)
    word_type_result = word_type(word)
    wordnet_result = wordnet_analysis(wordnet_dataset, word)

    return (vowel, harmony, word_info, etymology, word_type_result, wordnet_result)


def pMw(long_text):
    vowel_list = []
    harmony_list = []
    wordinfo_list = []
    etymology_list = []
    type_list = []
    wordnet_list = []

    with ThreadPoolExecutor() as execute:
        results = list(execute.map(pw, long_text))

    for result in results:
        vowel, harmony, word_info, etymology, word_type_result, wordnet_result = result
        vowel_list.append(vowel)
        harmony_list.append(harmony)
        if word_info:
            wordinfo_list.append(word_info)
        if etymology:
            etymology_list.append(etymology)
        if word_type_result:
            type_list.append(word_type_result)
        if wordnet_result:
            wordnet_list.append(wordnet_result)

    syllableharmony_tt = "  ".join(harmony_list)
    vowelextranctor_tt = " | ".join(vowel_list)
    wordtype_tt = " | ".join(type_list)
    etymology_tt = " | ".join(etymology_list)
    wordnet_tt = " | ".join(wordnet_list)
    wordinfo_tt = " | ".join(wordinfo_list) if wordinfo_list else None

    return (
        syllableharmony_tt,
        vowelextranctor_tt,
        wordtype_tt,
        etymology_tt,
        wordnet_tt,
        wordinfo_tt,
    )


@network.route("/", methods=["POST", "GET"])
def index():

    uyarı = phonetic_tt = rhyme_tt = wordsentiment_tt = textsentiment_tt = (
        sentimentgraph_tt
    ) = None
    syllabification_tt = vowelextranctor_tt = syllableharmony_tt = etymology_tt = (
        morphology_tt
    ) = None
    uyarı_rhyme = wordtype_tt = sesolayı_tt = plurality_tt = wordnet_tt = (
        wordinfo_tt
    ) = None

    if request.method == "POST":

        form_type = request.form["form_type"]
        user_input = request.form["text"]

        user_input_deasciified = deasciifier_function_F(user_input)
        user_input_deasciified = user_input_deasciified.strip().lower()
        long_text = wt(user_input_deasciified)
        long_length_text = len(long_text) > 20
        is_single_word = len(long_text) == 1

        if form_type == "general":
            if long_length_text:
                uyarı = "-- Lütfen en fazla 20 sözcük yazınız. --"

            else:
                phonetic_tt = phonetic_analysis(user_input_deasciified)
                if is_single_word:
                    plurality_tt = plurality_analysis(user_input_deasciified)
                    sesolayı_tt = sound_event(user_input_deasciified)
                    syllabification_tt = " | ".join(syll(user_input_deasciified))
                    vowelextranctor_tt = vowel_extractor(user_input_deasciified)
                    syllableharmony_tt = syllable_harmony(user_input_deasciified)

                    morphology_result = morphology_analysis(user_input_deasciified)
                    if morphology_result:
                        morphology_tt = morphology_result

                    word_sentiment_result = word_sentiment_analysis(
                        sentiment_dataset, user_input_deasciified
                    )
                    if word_sentiment_result:
                        wordsentiment_tt = word_sentiment_result
                        sentimentgraph_tt = sentiment_graph_generator(
                            word_sentiment_result
                        )

                    wordinfo_result = word_information(user_input_deasciified)
                    if wordinfo_result:
                        wordinfo_tt = wordinfo_result

                    etymology_result = etymology_analysis(user_input_deasciified)
                    if etymology_result:
                        etymology_tt = etymology_result

                    wordtype_result = word_type(user_input_deasciified)
                    if wordtype_result:
                        wordtype_tt = wordtype_result

                    wordnet_result = wordnet_analysis(
                        wordnet_dataset, user_input_deasciified
                    )
                    if wordnet_result:
                        wordnet_tt = wordnet_result

                else:
                    (
                        syllableharmony_tt,
                        vowelextranctor_tt,
                        wordtype_tt,
                        etymology_tt,
                        wordnet_tt,
                        wordinfo_tt,
                    ) = pMw(long_text)

                    text_sentiment_result = text_sentiment_analysis(
                        sentiment_dataset, user_input_deasciified
                    )
                    if text_sentiment_result != 0:
                        textsentiment_tt = text_sentiment_result
                        sentimentgraph_tt = sentiment_graph_generator(textsentiment_tt)

        elif form_type == "rhyme_tt":
            if is_single_word:
                rhyme_result = rhyme_generator(user_input_deasciified)
                if len(rhyme_result) > 0:
                    rhyme_tt = " | ".join(rhyme_result)
                else:
                    uyarı_rhyme = "-- Kafiye sözcüğü bulunamamıştır. --"
            else:
                uyarı_rhyme = "-- Lütfen bir tane sözcük yazınız. --"

    return render_template(
        "index.html",
        uyarı=uyarı,
        uyarı_rhyme=uyarı_rhyme,
        phonetic_tt=phonetic_tt,
        rhyme_tt=rhyme_tt,
        plurality_tt=plurality_tt,
        morphology_tt=morphology_tt,
        sesolayı_tt=sesolayı_tt,
        syllabification_tt=syllabification_tt,
        wordsentiment_tt=wordsentiment_tt,
        textsentiment_tt=textsentiment_tt,
        vowelextranctor_tt=vowelextranctor_tt,
        wordinfo_tt=wordinfo_tt,
        etymology_tt=etymology_tt,
        syllableharmony_tt=syllableharmony_tt,
        wordtype_tt=wordtype_tt,
        wordnet_tt=wordnet_tt,
        sentimentgraph_tt=sentimentgraph_tt,
    )


@network.route("/about")
def about():
    return render_template("about.html")


@network.route("/contact")
def contact():
    return render_template("contact.html")
