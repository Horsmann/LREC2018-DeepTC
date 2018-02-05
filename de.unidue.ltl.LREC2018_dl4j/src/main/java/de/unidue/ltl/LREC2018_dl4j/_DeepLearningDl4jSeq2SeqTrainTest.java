/**
 * Copyright 2017
 * Ubiquitous Knowledge Processing (UKP) Lab
 * Technische Universität Darmstadt
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */
package de.unidue.ltl.LREC2018_dl4j;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.CollectionReaderFactory.createReaderDescription;
import static org.dkpro.tc.core.DeepLearningConstants.DIM_PRETRAINED_EMBEDDINGS;
import static org.dkpro.tc.core.DeepLearningConstants.DIM_USER_CODE;
import static org.dkpro.tc.core.DeepLearningConstants.DIM_VECTORIZE_TO_INTEGER;

import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import org.apache.uima.collection.CollectionReaderDescription;
import org.dkpro.lab.Lab;
import org.dkpro.lab.task.Dimension;
import org.dkpro.lab.task.ParameterSpace;
import org.dkpro.tc.core.Constants;
import org.dkpro.tc.ml.DeepLearningExperimentTrainTest;
import org.dkpro.tc.ml.deeplearning4j.Deeplearning4jAdapter;

import de.tudarmstadt.ukp.dkpro.core.io.tei.TeiReader;
import de.unidue.ltl.LREC2018.SequenceOutcomeAnnotator;

/**
 * This a pure Java-based experiment setup of POS tagging as sequence tagging.
 */
public class _DeepLearningDl4jSeq2SeqTrainTest implements Constants {
	public static final String LANGUAGE_CODE = "en";

	static String train = null;
	static String test = null;
	static String embedding = null;

	/**
	 * Runs the DL4J demo. The demo is runnable as-is and uses a small subset of
	 * the Brown corpus (not the WSJ, i.e. license restrictions) as provided train /
	 * test data. An embeddings dummy based on the GloVe vectors with only 250 entries is used.
	 * The data size is too small for any meaningful results. This
	 * setup demonstrates only the usage of DKPro TC.
	 */
	public static void main(String[] args) throws Exception {

		/* Set configuration in run.config */
    	String config="src/main/resources/run.config";

		Properties p = new Properties();
		FileInputStream f = new FileInputStream(config);
		p.load(f);
		f.close();

		embedding = p.getProperty("embedding").trim();
		train = p.getProperty("train").trim();
		test = p.getProperty("test").trim();

		System.setProperty("DKPRO_HOME", System.getProperty("user.home") + "/Desktop");

		runExperiment(train, test, embedding);
	}

	public static void runExperiment(String trainData, String testData, String embedding)
			throws Exception {
		
		CollectionReaderDescription trainReader = createReaderDescription(
				TeiReader.class,
				TeiReader.PARAM_LANGUAGE, "en", 
				TeiReader.PARAM_SOURCE_LOCATION, train,
				TeiReader.PARAM_PATTERNS, "*.xml");

		CollectionReaderDescription testReader = createReaderDescription(
				TeiReader.class,
				TeiReader.PARAM_LANGUAGE, "en", 
				TeiReader.PARAM_SOURCE_LOCATION, test,
				TeiReader.PARAM_PATTERNS, "**/*.xml");
		
		/* WSJ data format reader */
//		CollectionReaderDescription trainReader = createReaderDescription(
//				PennTreebankChunkedReader.class,
//				PennTreebankChunkedReader.PARAM_LANGUAGE, "en", 
//				PennTreebankChunkedReader.PARAM_SOURCE_LOCATION, train,
//				PennTreebankChunkedReader.PARAM_PATTERNS, "**/*.pos");
//
//		CollectionReaderDescription testReader = createReaderDescription(
//				PennTreebankChunkedReader.class,
//				PennTreebankChunkedReader.PARAM_LANGUAGE, "en", 
//				PennTreebankChunkedReader.PARAM_SOURCE_LOCATION, test,
//				PennTreebankChunkedReader.PARAM_PATTERNS, "**/*.pos");
		
        Map<String, Object> dimReaders = new HashMap<String, Object>();
		dimReaders.put(DIM_READER_TRAIN, trainReader);
		dimReaders.put(DIM_READER_TEST, testReader);

		ParameterSpace pSpace = new ParameterSpace(
				Dimension.createBundle("readers", dimReaders),
				Dimension.create(DIM_FEATURE_MODE, FM_SEQUENCE),
				Dimension.create(DIM_LEARNING_MODE, LM_SINGLE_LABEL),
				Dimension.create(DIM_PRETRAINED_EMBEDDINGS, embedding),
				Dimension.create(DIM_VECTORIZE_TO_INTEGER, false),
				Dimension.create(DIM_USER_CODE, new Dl4jSeq2SeqUserCode()));
		
		DeepLearningExperimentTrainTest exp = new DeepLearningExperimentTrainTest("Dl4jExperiment",
				Deeplearning4jAdapter.class);
		exp.setParameterSpace(pSpace);
		exp.setPreprocessing(createEngineDescription(SequenceOutcomeAnnotator.class));

		Lab.getInstance().run(exp);

	}
}
