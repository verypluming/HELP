
#!/usr/bin/python
# -*- coding: utf-8 -*-
#  Copyright 2019 Hitomi Yanaka
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import glob
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from lxml import etree
from nltk.corpus import wordnet as wn
import xml.dom.minidom
from pattern3.en import *
import subprocess
import copy
import os
import sys
from xml.dom import minidom
import inflect
import logging as log
from nltk.wsd import lesk
inflect = inflect.engine()


def keep_plurals(noun, newnoun):
    if inflect.singular_noun(noun) is False:
        # singular
        return singularize(newnoun)
    else:
        # plural
        return pluralize(newnoun)

def keep_tenses(verb, newverb):
    ori_tense = tenses(verb)[0]
    ori_tense2 = [x for x in ori_tense if x is not None]
    #print(ori_tense2)
    tense, person, number, mood, aspect = None, None, None, None, None

    if 'infinitive' in ori_tense2:
        tense = INFINITIVE
    elif 'present' in ori_tense2:
        tense = PRESENT
    elif 'past' in ori_tense2:
        tense = PAST
    elif 'future' in ori_tense2:
        tense = FUTURE

    if 1 in ori_tense2:
        person = 1
    elif 2 in ori_tense2:
        person = 2
    elif 3 in ori_tense2:
        person = 3
    else:
        person = None

    if 'singular' in ori_tense2:
        number = SINGULAR
    elif 'plural' in ori_tense2:
        number = PLURAL
    else:
        number = None

    if 'indicative' in ori_tense2:
        mood = INDICATIVE
    elif 'imperative' in ori_tense2:
        mood = IMPERATIVE
    elif 'conditional' in ori_tense2:
        mood = CONDITIONAL
    elif 'subjunctive' in ori_tense2:
        mood = SUBJUNCTIVE
    else:
        mood = None
    
    #if 'imperfective' in ori_tense2:
    #   aspect = IMPERFECTIVE
    #elif 'perfective' in ori_tense2:
    #   aspect = PERFECTIVE
    if 'progressive' in ori_tense2:
        aspect = PROGRESSIVE
    else:
        aspect = None

    newverb_tense = conjugate(newverb, 
        tense = tense,        # INFINITIVE, PRESENT, PAST, FUTURE
       person = person,              # 1, 2, 3 or None
       number = number,       # SG, PL
         mood = mood,     # INDICATIVE, IMPERATIVE, CONDITIONAL, SUBJUNCTIVE
        aspect = aspect,
      negated = False,          # True or False
        parse = True)
    #print(newverb, newverb_tense)
    return newverb_tense

# functions for replacement
def remove_duplicates(x):
    y=[]
    for i in x:
        if i not in y:
            y.append(i)
    return y

def replace_sentence(determiner, nounmono, noun, newnoun, sentence, results, target):
    pat = re.compile(noun)
    newpat = re.compile(newnoun)
    newsentence = re.sub(noun, newnoun, sentence)
    gold_label = check_label(nounmono, 'simple')
    record = pd.Series([target, determiner, nounmono, gold_label, noun, newnoun, 'simple', sentence, newsentence], index=results.columns)
    record = pd.Series([target, determiner, nounmono, rev_label(gold_label, nounmono), noun, newnoun, 'simple', newsentence, sentence], index=results.columns)
    results = results.append(record, ignore_index = True)
    return results

def replace_sentence_WN_nv(determiner, nounmono, verbmono, noun, nounsense, verb, verbsense, sentence, results, target):
    nounsynset = wn.synset(nounsense)
    nounhypernyms = nounsynset.hypernyms()
    nounhyponyms = nounsynset.hyponyms()
    verbsynset = wn.synset(verbsense)
    verbhypernyms = verbsynset.hypernyms()
    verbhyponyms = verbsynset.hyponyms()
    
    nounhypersim = [nounhypernym.wup_similarity(verbsynset) if nounhypernym.wup_similarity(verbsynset) is not None else 0 for nounhypernym in nounhypernyms]
    nounhyposim = [nounhyponym.wup_similarity(verbsynset) if nounhyponym.wup_similarity(verbsynset) is not None else 0 for nounhyponym in nounhyponyms]
    verbhypersim = [verbhypernym.wup_similarity(nounsynset) if verbhypernym.wup_similarity(nounsynset) is not None else 0 for verbhypernym in verbhypernyms]
    verbhyposim = [verbhyponym.wup_similarity(nounsynset) if verbhyponym.wup_similarity(nounsynset) is not None else 0 for verbhyponym in verbhyponyms]
    
    nounhypernym = nounhypernyms[nounhypersim.index(max(nounhypersim))]
    nounhyponym = nounhyponyms[nounhyposim.index(max(nounhyposim))]
    verbhypernym = verbhypernyms[verbhypersim.index(max(verbhypersim))]
    verbhyponym = verbhyponyms[verbhyposim.index(max(verbhyposim))]

    synsetdict = {#"noun_synset": nounsynset,
                  "noun_hypernym": nounhypernym,
                  "noun_hyponym": nounhyponym,
                  #"verb_synset": verbsynset,
                  "verb_hypernym": verbhypernym,
                  "verb_hyponym": verbhyponym
                 }
    #print(synsetdict)
    for rel, synset in synsetdict.items():
        synsetwords = synset.lemma_names()
        #print(synsetwords)
        for synsetword in synsetwords:
            new_synsetword = re.sub("_", " ", synsetword)
            if re.search("noun", rel):
                newnoun = keep_plurals(noun, new_synsetword)
                pat = re.compile(noun)
                newpat = re.compile(newnoun)
                newsentence = re.sub(noun, newnoun, sentence)
                gold_label = check_label(nounmono, rel)
                record = pd.Series([target, determiner, nounmono, gold_label, noun, newnoun, rel, sentence, newsentence], index=results.columns)
                results = results.append(record, ignore_index = True)
                record = pd.Series([target, determiner, nounmono, rev_label(gold_label, nounmono), noun, newnoun, rel, newsentence, sentence], index=results.columns)
                results = results.append(record, ignore_index = True)
            else:
                newverb = keep_tenses(verb, new_synsetword)
                pat = re.compile(verb)
                newpat = re.compile(newverb)
                newsentence = re.sub(verb, newverb, sentence)
                gold_label = check_label(verbmono, rel)
                record = pd.Series([target, determiner, verbmono, gold_label, verb, newverb, rel, sentence, newsentence], index=results.columns)
                results = results.append(record, ignore_index = True)
                record = pd.Series([target, determiner, verbmono, rev_label(gold_label, verbmono), verb, newverb, rel, newsentence, sentence], index=results.columns)
                results = results.append(record, ignore_index = True)

    return results

def replace_sentence_WN(determiner, nounmono, noun, sense, sentence, results, target):
    synset = wn.synset(sense)
    hypernyms = synset.hypernyms()
    hyponyms = synset.hyponyms()

    for hypernym in hypernyms:
        #if len(hypernym.examples()) == 0:
        #    # remove if no example exists in WordNet
        #    continue
        hypernym_words = hypernym.lemma_names()
        for hypernym_word in hypernym_words:
            #print(hypernym_word)
            new_hypernym_word = re.sub("_", " ", hypernym_word)
            newnoun = keep_plurals(noun, new_hypernym_word)
            pat = re.compile(noun)
            newpat = re.compile(newnoun)
            newsentence = re.sub(noun, newnoun, sentence)
            gold_label = check_label(nounmono, 'noun_hypernym_obj')
            record = pd.Series([target, determiner, nounmono, gold_label, noun, newnoun, 'noun_hypernym_obj', sentence, newsentence], index=results.columns)
            results = results.append(record, ignore_index = True)
            record = pd.Series([target, determiner, nounmono, rev_label(gold_label, nounmono), noun, newnoun, 'noun_hypernym_obj', newsentence, sentence], index=results.columns)
            results = results.append(record, ignore_index = True)
            
    for hyponym in hyponyms:
        #if len(hyponym.examples()) == 0:
        #    # remove if no example exists in WordNet
        #    continue
        hyponym_words = hyponym.lemma_names()
        for hyponym_word in hyponym_words:
            #print(hyponym_word)
            new_hyponym_word = re.sub("_", " ", hyponym_word)
            newnoun = keep_plurals(noun, new_hyponym_word)
            pat = re.compile(noun)
            newpat = re.compile(newnoun)
            newsentence = re.sub(noun, newnoun, sentence)
            gold_label = check_label(nounmono, 'noun_hyponym_obj')
            record = pd.Series([target, determiner, nounmono, gold_label, noun, newnoun, 'noun_hyponym_obj', sentence, newsentence], index=results.columns)
            results = results.append(record, ignore_index = True)
            record = pd.Series([target, determiner, nounmono, rev_label(gold_label, nounmono), noun, newnoun, 'noun_hypernym_obj', newsentence, sentence], index=results.columns)
            results = results.append(record, ignore_index = True)
    return results

def replace_sentence_numeral(det, num, sentence, results, target):
    #not used
    tmpnum = str(number(num))
    tmpnum = re.sub(",", "", tmpnum)
    if det.lower() in ['more', 'greater', 'larger', 'taller', 'bigger', 'least']:
        #upward monotonicity
        pat = re.compile(num)
        newnum = str(int(tmpnum) - 1) #trial
        newpat = re.compile(newnum)
        newsentence = re.sub(num, newnum, sentence)
        record = pd.Series([target, num, newnum, 'numeral', sentence, newsentence], index=results.columns)
        #print(target, newnum, newsentence)
        results = results.append(record, ignore_index = True)
    elif det.lower() in ['less', 'fewer', 'smaller', 'shorter', 'most']:
        #downward monotonicity
        pat = re.compile(num)
        newnum = str(int(tmpnum) + 1) #trial
        newpat = re.compile(newnum)
        newsentence = re.sub(num, newnum, sentence)
        record = pd.Series([target, num, newnum, 'numeral', sentence, newsentence], index=results.columns)
        results = results.append(record, ignore_index = True)
    else:
        print("target: "+target+", other determiner: "+determiner) 
    return results

#candc2transccg
def get_nodes_by_tag(root, tag):
    nodes = []
    if root.tag == tag:
        nodes.append(root)
    for node in root:
        nodes.extend(get_nodes_by_tag(node, tag))
    return nodes

def assign_ids_to_nodes(ccg_tree, sentence_number, current=0):
    ccg_tree.set('id', 's' + str(sentence_number) + '_sp' + str(current))
    current += 1
    for node in ccg_tree:
        current = assign_ids_to_nodes(node, sentence_number, current)
    return current

def rename_attributes(ccg_root, src_attribute, trg_attribute):
    if src_attribute in ccg_root.attrib:
        ccg_root.set(trg_attribute, ccg_root.get(src_attribute))
        del ccg_root.attrib[src_attribute]
    for child_node in ccg_root:
        rename_attributes(child_node, src_attribute, trg_attribute)

def assign_values_in_feat_structs(ccg_root):
    assert 'category' in ccg_root.attrib, 'Category field not present in node {0}'\
      .format(etree.tostring(ccg_root, pretty_print=True))
    category = ccg_root.get('category')
    category_assigned_value = re.sub(r'([,\]])', r'=true\1', category)
    ccg_root.set('category', category_assigned_value)
    for child_node in ccg_root:
        assign_values_in_feat_structs(child_node)

def assign_child_info(ccg_tree, sentence_number, tokens_node):
    if len(ccg_tree) == 0:
        token_position = ccg_tree.get('start')
        ccg_tree.set('terminal', 't' + str(sentence_number) + '_' + str(token_position))
    else:
        child_str = ' '.join([child_node.get('id') for child_node in ccg_tree])
        ccg_tree.set('child', child_str)
        ccg_tree.set('pos', "None")
    for child_node in ccg_tree:
        assign_child_info(child_node, sentence_number, tokens_node)

def flatten_and_rename_nodes(ccg_root):
    spans = []
    ccg_root.tag = 'span'
    spans.append(ccg_root)
    for child_node in ccg_root:
        spans.extend(flatten_and_rename_nodes(child_node))
    return spans

def candc_to_transccg(ccg_tree, sentence_number):
    # Obtain the <tokens> subtree and store it in variable tokens_node.
    tokens = get_nodes_by_tag(ccg_tree, 'lf')
    for i, token in enumerate(tokens):
        token.tag = 'token'
        token.set('id', 't' + str(sentence_number) + '_' + str(i))
        # Prefix every surface and base form with an underscore.
        # This is useful to avoid collisions of reserved words (e.g. "some", "all")
        # in nltk or coq. We also substitute dots '.' by 'DOT'.
        word = normalize_string(token.get('word'), 'surf')
        lemma = normalize_string(token.get('lemma'), 'base')
        token.set('surf', word)
        token.set('base', lemma)
        del token.attrib['word']
        del token.attrib['lemma']
    tokens_node = etree.Element('tokens')
    for token in tokens:
        tokens_node.append(copy.deepcopy(token))
    # Obtain the <ccg> subtree and store it in variable ccg_node.
    ccg_tree.set('root', 's' + str(sentence_number) + '_sp0')
    ccg_tree.set('id', 's' + str(sentence_number) + '_ccg0')
    # Assign an ID to every node, in depth order.
    ccg_root = ccg_tree[0]
    ccg_root.set('root', 'true')
    assign_ids_to_nodes(ccg_root, sentence_number)
    assign_child_info(ccg_root, sentence_number, tokens_node)
    # Rename attributes.
    rename_attributes(ccg_root, 'cat', 'category')
    rename_attributes(ccg_root, 'type', 'rule')
    # Assign values to feature structures. E.g. S[adj] --> S[adj=true]
    assign_values_in_feat_structs(ccg_root)
    # Flatten structure.
    spans = flatten_and_rename_nodes(ccg_root)
    for child_span in spans:
        ccg_tree.append(child_span)
        if child_span.get('id').endswith('sp0'):
            child_span.set('root', 'true')
    sentence_node = etree.Element('sentence')
    sentence_node.append(tokens_node)
    sentence_node.append(ccg_tree)
    return sentence_node

def normalize_string(raw_string, attribute):
    normalized = raw_string
    if attribute == 'base':
        normalized = normalized.lower()
    return normalized

def make_transccg_xml_tree(transccg_trees):
    """
    Create the structure:
    <root>
      <document>
        <sentences>
          <sentence id="s1">
          ...
          </sentence>
        </sentences>
      </document>
    </root>
    """
    sentences_node = etree.Element('sentences')
    for transccg_tree in transccg_trees:
        sentences_node.append(transccg_tree)
    document_node = etree.Element('document')
    document_node.append(sentences_node)
    root_node = etree.Element('root')
    root_node.append(document_node)
    return root_node

def candc2transccg(candc_trees):
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(candc_trees, parser)
    #root = xml_tree.getroot()
    ccg_trees = root.findall('ccg')

    transccg_trees = []
    for i, ccg_tree in enumerate(ccg_trees):
        transccg_tree = candc_to_transccg(ccg_tree, i)
        transccg_trees.append(transccg_tree)

    transccg_xml_tree = make_transccg_xml_tree(transccg_trees)
    # transccg_xml_tree.write(pretty_print=True, encoding='utf-8')
    parse_result = etree.tostring(transccg_xml_tree, xml_declaration=True, pretty_print=True)
    return parse_result

def parse(parser_name, sentence):
    parse_result = ""
    f = open("../data/parser_location.txt","r")
    locations = f.readlines()
    f.close()
    candc_dir = locations[0].split(":")[1].strip()
    if parser_name == "candc":
        # Parse using C&C.
        command = "echo "+sentence+"|"+candc_dir+"bin/candc --models "+candc_dir+"models --candc-printer xml"
        result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = result.communicate()
        parse_result = candc2transccg(out)
    return parse_result

def check_monotonicity(determiner):
    nounmono, verbmono = "non_monotone", "non_monotone"
    upward_noun = ["some", "a"]
    upward_verb = ["every", "each", "all", "some", "both", "most", "many", "several", "this", "that", "a", "the"]
    downward_noun = ["every", "each", "all", "no", "neither", "any", "never"]
    downward_verb = ["no", "neither", "any", "never", "few"]
    if determiner in upward_noun:
        nounmono = "upward_monotone"
    if determiner in upward_verb:
        verbmono = "upward_monotone"
    if determiner in downward_noun:
        nounmono = "downward_monotone"
    if determiner in downward_verb:
        verbmono = "downward_monotone"
    return nounmono, verbmono

def check_label(monotonicity, mode):
    modegroup = ""
    if re.search("hypo", mode):
        modegroup = "down"
    elif re.search("hyper", mode):
        modegroup = "up"
    elif mode == "simple":
        modegroup = "up"
    if monotonicity == "upward_monotone" and modegroup == "up":
        return "entailment"
    elif monotonicity == "upward_monotone" and modegroup == "down":
        return "neutral"
    elif monotonicity == "downward_monotone" and modegroup == "up":
        return "neutral"
    elif monotonicity == "downward_monotone" and modegroup == "down":
        return "entailment"
    else:
        return "neutral"

def rev_label(gold_label, monotonicity):
    #reverse the gold_label
    if monotonicity == "non_monotone":
        return "neutral"
    elif gold_label == "entailment":
        return "neutral"
    elif gold_label == "neutral":
        return "entailment"

def rev_mono(monotonicity):
    #reverse the polarity
    if monotonicity == "non_monotone":
        return "non_monotone"
    elif monotonicity == "downward_monotone":
        return "upward_monotone"
    elif monotonicity == "upward_monotone":
        return "downward_monotone"

def main():
    parser = etree.XMLParser(remove_blank_text=True)
    files = glob.glob("../data/pmb-2.1.0/data/*/*/*/en.drs.xml")
    #files = glob.glob("../data/pmb-2.1.0/data/gold/*/*/en.drs.xml")
    df_list = []
    determiners = ["every", "each", "all", "some", "no", "both", "neither", "most", "many", "any",\
    "several", "exactly"]
    #determiners = ["each"]
    floating_list = ["both", "all", "each"]
    #a ,the, this, that
    for determiner in determiners:
        target_files = []
        nounmono, verbmono = check_monotonicity(determiner)
        for file in files:
            filename = re.search("\/data\/pmb-2.1.0\/data\/(.*?)\/en.drs.xml", file).group(1)
            try:
                tree = etree.parse("../data/pmb-2.1.0/data/"+filename+"/en.drs.xml", parser)
                words = tree.xpath("//taggedtokens/tagtoken/tags/tag[@type='lemma']/text()")
                if determiner in words:
                    target_files.append(filename)
            except:
                continue

        results = pd.DataFrame(index=[], columns=['filename', 'determiner', 'monotonicity', 'gold_label', 'replace_target', 'replace_source', 'replace_mode', 'ori_sentence', 'new_sentence'])
        #target_files = ["silver/p47/d2720"]
        for target in target_files:
            #print(target)
            try:
                tree2 = etree.parse("../data/pmb-2.1.0/data/"+target+"/en.drs.xml", parser)
                #semtag = tree.xpath("//taggedtokens/tagtoken/tags/tag[@type='sem']/text()")
                #perid = tree.xpath("//taggedtokens/tagtoken/tags/tag[@type='sem'][contains(text(), 'PER')]/../../@xml:id")
                impid = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='sem'][contains(text(), 'IMP')]/../../@xml:id")
                negid = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='sem'][contains(text(), 'NOT')]/../../@xml:id")
                if len(negid) > 0 or len(impid) > 0:
                    # reverse monotonicity if negation or implication exists
                    print(target+": contains negation or implication. reverse monotonicity\n")
                    nounmono = rev_mono(nounmono)
                    verbmono = rev_mono(verbmono)
                queid = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='sem'][contains(text(), 'QUE')]/../../@xml:id")
                firstpos = tree2.xpath("//taggedtokens/tagtoken[@xml:id='i1001']/tags/tag[@type='pos']/text()")
                #tree2 = etree.parse("../data/pmb-2.1.0/data/gold/"+target+"/en.drs.xml", parser)
                IDs = len(tree2.xpath("//xdrs"))+1
                for ID in range(1, IDs):
                    floating_flg = 0
                    noun, newnoun, verb, newverb = "", "", "", ""
                    verbs = []
                    nouns = []
                    words = []
                    words = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='tok']/text()")
                    if len(words) <= 5:
                        # remove less than 5 words
                        print(target+": is less than 5 words\n")
                        continue
                    if len(firstpos) > 0:
                        meirei = firstpos[0]
                        if re.search("^VB", meirei):
                            # remove imperatives
                            print(target+": is meireikei\n")
                            continue
                    if "\"" in words or len(queid) > 0:
                        # remove questions
                        print(target+": contains quotation or question\n")
                        continue
                    sentence = " ".join(words)
                    sentence = re.sub("ø ", "", sentence)
                    if determiner == "no":
                        # remove collocations
                        if re.search("no one", sentence) or re.search("No one", sentence) or re.search("No doubt", sentence) or re.search("no doubt", sentence) or re.search("No ,", sentence):
                            continue
                    #print(sentence)
                    parse_result = parse("candc", sentence)
                    #doc = minidom.parseString(parse_result)
                    #print(doc.toxml())                
                    tree3 = etree.fromstring(parse_result, parser)
                    target_id = tree3.xpath("//ccg/span[@base='" + determiner + "']/@id")
                    verb_id = []
                    child_ids, child_verb_ids = [], []
                    #print(target_id)

                    # detect the parent node of NP and VP
                    while True:
                        parent_id = tree3.xpath("//ccg/span[contains(@child, '" + target_id[0] + "')]/@id")
                        parent_category = tree3.xpath("//ccg/span[contains(@child, '" + target_id[0] + "')]/@category")[0]
                        #print(parent_category)
                        if not re.search("^NP\[?", parent_category):
                            tmp4 = tree3.xpath("//ccg/span[contains(@child, '" + target_id[0] + "')]/@child")
                            if len(tmp4) > 0:
                                verb_id = tmp4[0].split(" ")
                                verb_id.remove(target_id[0])
                                verb_base =  tree3.xpath("//ccg/span[contains(@id, '" + verb_id[0] + "')]/@base")
                                if 'be' in verb_base and determiner in floating_list:
                                    #floating
                                    floating_flg = 1
                                break
                        else:
                            target_id = parent_id

                    #print(target_id, verb_id)
                    # extract the whole NP subtree
                    list_target_id = target_id[0].split(" ")
                    while True:
                        childid = []
                        for parentid in list_target_id:
                            tmp = tree3.xpath("//ccg/span[contains(@id, '" + parentid + "')]/@child")
                            if len(tmp) > 0:
                                childid.extend(tmp[0].split(" "))
                        if len(childid) == 0:
                            break
                        else:
                            child_ids.extend(childid)
                            list_target_id = childid
                    
                    # extract the whole VP subtree
                    list_verb_id = verb_id[0].split(" ")
                    while True:
                        childid = []
                        for parentid in list_verb_id:
                            tmp5 = tree3.xpath("//ccg/span[contains(@id, '" + parentid + "')]/@child")
                            if len(tmp5) > 0:
                                childid.extend(tmp5[0].split(" "))
                        if len(childid) == 0:
                            break
                        else:
                            child_verb_ids.extend(childid)
                            list_verb_id = childid

                    for nounphrase in sorted(child_ids, key=lambda x:int((re.search(r"sp([0-9]+)", x)).group(1))):
                        tmp2 = tree3.xpath("//ccg/span[@id='" + nounphrase + "']/@surf")
                        if len(tmp2) > 0:
                            nouns.extend(tmp2)
                    #print(nounphrase, nouns)
                    
                    for verbphrase in sorted(child_verb_ids, key=lambda x:int((re.search(r"sp([0-9]+)", x)).group(1))):
                        tmp3 = tree3.xpath("//ccg/span[@id='" + verbphrase + "']/@surf")
                        if len(tmp3) > 0:
                            verbs.extend(tmp3)

                    if floating_flg == 1:
                        # remove floating
                        continue
                    # replace an subjective word by its hypernym and hyponym
                    elif len(nouns) > 0 and len(verbs) > 0:
                        noun = " ".join(nouns)
                        newnoun = nouns[-1]
                        newnounpos = tree3.xpath("//ccg/span[@surf='" + newnoun + "']/@pos")[0]
                        if re.search("^PRP", newnounpos):
                            # remove pronouns
                            print(target+": is pronoun\n")
                            continue
                        if re.search("^NNP", newnounpos):
                            # replace its specific hypernym if a proper noun exists
                            print(target+" contains koyumeishi\n")
                            semtag = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='tok' and text()='" + newnoun + "']/following-sibling::tag[@type='sem']/text()")
                            if len(semtag) > 0:
                                if semtag[0] == "PER" or semtag[0] == "GPO":
                                    newnoun = "someone"
                                elif semtag[0] == "GPE" or semtag[0] == "GEO":
                                    newnoun = "somewhere"
                                else:
                                    print(target+" contains other semtag"+semtag[0]+"\n")
                                    newnoun = "something"
                                results = replace_sentence(determiner, nounmono, noun, newnoun, sentence, results, target)
                                continue
                        if len(nouns) > 2:
                            newnewnoun = determiner + " " + nouns[-1]
                            results = replace_sentence(determiner, nounmono, noun, newnewnoun, sentence, results, target)
                        verb = " ".join(verbs)
                        newverb = verbs[-1]
                        #print(results)
                        # replace hypernym and hyponym using senseid
                        nounsense = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='tok' and text()='" + newnoun + "']/following-sibling::tag[@type='wordnet']/text()")
                        verbsense = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='tok' and text()='" + newverb + "']/following-sibling::tag[@type='wordnet']/text()")
                        if nounsense[0] == 'O':
                            nounsense = [str(lesk(words, newnoun, 'n'))[8:-2]]
                        if verbsense[0] == 'O':
                            verbsense = [str(lesk(words, newverb, 'v'))[8:-2]]
                        results = replace_sentence_WN_nv(determiner, nounmono, verbmono, newnoun, nounsense[0], newverb, verbsense[0], sentence, results, target)
                
                    elif len(nouns) > 0:
                        # replace an objective word by its hypernym and hyponym
                        noun = " ".join(nouns)
                        newnoun = nouns[-1] 
                        if len(nouns) > 2:
                            newnewnoun = determiner + " " + nouns[-1]
                            results = replace_sentence(determiner, nounmono, noun, newnewnoun, sentence, results, target)
                            #print(results)
                        # replace hypernym and hyponym using senseid
                        nounsense = tree2.xpath("//taggedtokens/tagtoken/tags/tag[@type='tok' and text()='" + newnoun + "']/following-sibling::tag[@type='wordnet']/text()")
                        if nounsense[0] == 'O':
                            nounsense = [str(lesk(words, newnoun, 'n'))[8:-2]]
                        results = replace_sentence_WN(determiner, nounmono, newnoun, nounsense[0], sentence, results, target)

            except Exception as e:
                log.exception("ERROR target: "+target)
                log.exception(e)
                continue
        results.to_csv('../output_en/leskexppmb_'+determiner+'.tsv', sep='\t')



def format_files():
    datas = glob.glob("../output_en/leskexppmb_*.tsv")
    alldata = pd.DataFrame(index=[], columns=['filename', 'determiner', 'monotonicity', 'gold_label', 'replace_target', 'replace_source', 'replace_mode', 'ori_sentence', 'new_sentence'])

    for d in datas:
        dataframe = pd.read_csv(d, sep="\t", index_col=0)
        if len(dataframe) > 0:
            alldata = alldata.append(dataframe)

    
    alldata.to_csv("../output_en/pmb_train.tsv", sep='\t', index=False)
    
    # MultiNLI train format
    results = pd.DataFrame(index=[], columns=['index','promptID','pairID','genre','sentence1_binary_parse','sentence2_binary_parse','sentence1_parse','sentence2_parse','sentence1','sentence2','label1','gold_label'])
    results['label1'] = alldata['gold_label']
    results['gold_label'] = alldata['gold_label']
    results['sentence1'] = alldata['ori_sentence']
    results['sentence2'] = alldata['new_sentence']
    results['index'] = results.reset_index()
    results['promptID'] = results.reset_index()
    results['pairID'] = results.reset_index()
    results.to_csv("../output_en/pmb_train_mnliformat.tsv", sep="\t", index=False, header=False)



if __name__ == '__main__':
    main()
    format_files()
    
