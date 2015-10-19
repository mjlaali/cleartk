/* 
 * Copyright (c) 2012, Regents of the University of Colorado 
 * All rights reserved.
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * For a complete copy of the license please see the file LICENSE distributed 
 * with the cleartk-syntax-berkeley project or visit 
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html.
 */

package org.cleartk.berkeleyparser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.uima.cas.CASException;
import org.apache.uima.jcas.cas.FSArray;
import org.cleartk.syntax.constituent.type.TerminalTreebankNode;
import org.cleartk.syntax.constituent.type.TopTreebankNode;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;

import edu.berkeley.nlp.io.LineLexer;

/**
 * <br>
 * Copyright (c) 2010, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Majid Laali
 */
public class DefaultBerkeleyTokenizer implements Tokenizer<Token, Sentence, TopTreebankNode> {
  private LineLexer tokenizer = new LineLexer();

  public List<Token> tokenize(Sentence sent) {
    try {
      int base = sent.getBegin();
      String strSent = sent.getCoveredText();
      List<String> strTokens = tokenizer.tokenizeLine(strSent);
      List<Token> tokens = new ArrayList<>();
      
      int index = 0;
      for (String strToken: strTokens){
        index = strSent.indexOf(strToken, index);
        if (index == -1)
          throw new RuntimeException(String.format("Cannot find token <%s> in the sentence <%s>: ", 
              strToken, strSent));
        Token token = new Token(sent.getCAS().getJCas(), base + index, base + index + strToken.length());
        token.addToIndexes();
        tokens.add(token);
        index += strToken.length();
      }
      return tokens;
    } catch (IOException e) {
      e.printStackTrace();
    } catch (CASException e) {
      e.printStackTrace();
    }
    return null;
    
  }
  
  public void setPosTags(List<Token> tokens, TopTreebankNode topTreebankNode) {
    FSArray terminals = topTreebankNode.getTerminals();
    for (int i = 0; i < terminals.size(); i++){
      TerminalTreebankNode terminalTreebankNode = (TerminalTreebankNode) terminals.get(i);
      tokens.get(i).setPos(terminalTreebankNode.getParent().getNodeType());
    }
  }
}
