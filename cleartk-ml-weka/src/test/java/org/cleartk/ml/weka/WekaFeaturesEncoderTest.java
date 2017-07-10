/** 
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
package org.cleartk.ml.weka;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.List;

import org.cleartk.ml.Feature;
import org.cleartk.ml.encoder.CleartkEncoderException;
import org.junit.Before;
import org.junit.Test;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * Copyright (c) 2012, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Majid Laali
 */
public class WekaFeaturesEncoderTest {
  private Instances instances;
  private WekaFeaturesEncoder wekaFeatureEncoder;
  private WekaNominalFeatureEncoder wekaOutcomeEncoder;
  
  @Before
  public void init() throws CleartkEncoderException{
    List<List<Feature>> featureLists = Arrays.asList(
        Arrays.asList(new Feature[]{new Feature("F1", "V1")}),
        Arrays.asList(new Feature[]{new Feature("F1", "V3")}),
        Arrays.asList(new Feature[]{new Feature("F1", "V2")}),
        Arrays.asList(new Feature[]{new Feature("F1", "V1")})
        );

    List<String> outcomes = Arrays.asList("c1", "c2", "c1", "c2");

    wekaFeatureEncoder = new WekaFeaturesEncoder();
    wekaOutcomeEncoder = new WekaNominalFeatureEncoder("outcome", true);

    for (int i = 0; i < featureLists.size(); i++){
      wekaFeatureEncoder.encodeAll(featureLists.get(i));
      wekaOutcomeEncoder.encode(outcomes.get(i));
    }


    Attribute outcomeAttribute = wekaOutcomeEncoder.getAttribute();
    instances = wekaFeatureEncoder.makeInstances(featureLists.size(), outcomeAttribute, "relTag");

    for (int i = 0; i < featureLists.size(); i++){
      SparseInstance instance = wekaFeatureEncoder.createInstance(featureLists.get(i));
      wekaOutcomeEncoder.setAttributeValue(instance, outcomes.get(i));
      instances.add(instance);
    }
  }

  @Test
  public void whenTwoFeaturesAreTheSameThenAfterEncodingTheirValueAreTheSame(){
    assertThat(instances.numAttributes()).isEqualTo(2);
    
    Attribute featureAttr = instances.attribute("F1");
    assertThat(featureAttr.numValues()).isEqualTo(3 + 1); // +1 because of not seen featuer value
    for (int i = 1; i <= 3; i++)
      assertThat(featureAttr.indexOfValue("V" + i)).isEqualTo(i);
  }

  @Test
  public void givenALoadedEncoderWhenSaveANewValueThenItConvertToTheSameNominalIndexValue() throws IOException, ClassNotFoundException{
    ByteArrayOutputStream buff = new ByteArrayOutputStream();
    ObjectOutputStream io = new ObjectOutputStream(buff);
    io.writeObject(wekaFeatureEncoder);
    io.writeObject(wekaOutcomeEncoder);
    io.close();
    
    
    ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(buff.toByteArray()));
    WekaFeaturesEncoder encoder = (WekaFeaturesEncoder) input.readObject();
    WekaNominalFeatureEncoder outcomeEncoder = (WekaNominalFeatureEncoder) input.readObject();
    
    Attribute outcomeAttribute = outcomeEncoder.getAttribute();
    
    Instances testInstances = encoder.makeInstances(1, outcomeAttribute, "relTag");
    
    assertThat(testInstances.numAttributes()).isEqualTo(2);
    
    Attribute featureAttr = testInstances.attribute("F1");
    assertThat(featureAttr.numValues()).isEqualTo(3 + 1); // +1 because of not seen featuer value
    for (int i = 1; i <= 3; i++)
      assertThat(featureAttr.indexOfValue("V" + i)).isEqualTo(i);
  }

  @Test
  public void givenALoadedEncoderWhenCreatingTestInstancesThenTheGeneratedInstanceIsFine() throws IOException, ClassNotFoundException{
    ByteArrayOutputStream buff = new ByteArrayOutputStream();
    ObjectOutputStream io = new ObjectOutputStream(buff);
    io.writeObject(wekaFeatureEncoder);
    io.writeObject(wekaOutcomeEncoder);
    io.close();
    
    
    ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(buff.toByteArray()));
    WekaFeaturesEncoder encoder = (WekaFeaturesEncoder) input.readObject();
    WekaNominalFeatureEncoder outcomeEncoder = (WekaNominalFeatureEncoder) input.readObject();
    
    Attribute outcomeAttribute = outcomeEncoder.getAttribute();
    
    Instances testInstances = encoder.makeInstances(1, outcomeAttribute, "relTag");
    
    testInstances.add(encoder.createInstance(Arrays.asList(new Feature("F1", "V2"))));

    //classify the extracted instance.
    Instance wekaInstance = testInstances.instance(0);
    
    assertThat(wekaInstance.attribute(0).name()).isEqualTo("F1");
    assertThat(wekaInstance.value(0)).isEqualTo(2);
  }
}
