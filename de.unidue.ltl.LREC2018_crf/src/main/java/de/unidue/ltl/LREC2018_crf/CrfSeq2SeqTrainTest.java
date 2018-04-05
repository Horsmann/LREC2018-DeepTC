package de.unidue.ltl.LREC2018_crf;

import static java.util.Arrays.asList;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.CollectionReaderFactory.createReaderDescription;

import java.io.FileInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.resource.ResourceInitializationException;
import org.dkpro.lab.Lab;
import org.dkpro.lab.task.BatchTask.ExecutionPolicy;
import org.dkpro.lab.task.Dimension;
import org.dkpro.lab.task.ParameterSpace;
import org.dkpro.tc.api.features.TcFeatureFactory;
import org.dkpro.tc.api.features.TcFeatureSet;
import org.dkpro.tc.core.Constants;
import org.dkpro.tc.features.tcu.TargetSurfaceFormContextFeature;
import org.dkpro.tc.ml.ExperimentTrainTest;
import org.dkpro.tc.ml.crfsuite.CrfSuiteAdapter;
import org.dkpro.tc.ml.report.BatchRuntimeReport;

import de.tudarmstadt.ukp.dkpro.core.io.tei.TeiReader;
import de.unidue.ltl.LREC2018.SequenceOutcomeAnnotator;

public class CrfSeq2SeqTrainTest implements Constants {
	public static final String LANGUAGE_CODE = "en";

	/**
	 * Runs the CRF demo. The demo is runnable as-is and uses a small subset of
	 * the Brown corpus (not the WSJ, i.e. license restrictions) as provided train /
	 * test data. The data size is too small for any meaningful results. This
	 * setup demonstrates only the usage of DKPro TC.
	 */
	public static void main(String[] args) throws Exception {

		System.setProperty("DKPRO_HOME", System.getProperty("user.home") + "/Desktop");

		/* Set configuration in run.config */
    	String config="src/main/resources/run.config";


		Properties p = new Properties();
		FileInputStream f = new FileInputStream(config);
		p.load(f);
		f.close();

		String brownCluster = p.getProperty("brown").trim();
		String train = p.getProperty("train").trim();
		String test = p.getProperty("test").trim();
    	
		ParameterSpace pSpace = getParameterSpace(train, test, brownCluster);

		CrfSeq2SeqTrainTest experiment = new CrfSeq2SeqTrainTest();
		experiment.runTrainTest(pSpace);
	}

	public static ParameterSpace getParameterSpace(String train, String test, String brown)
			throws ResourceInitializationException {

        Map<String, Object> crf = new HashMap<>();
        crf.put(DIM_CLASSIFICATION_ARGS, new Object[] { new CrfSuiteAdapter(),
                CrfSuiteAdapter.ALGORITHM_ADAPTIVE_REGULARIZATION_OF_WEIGHT_VECTOR });
        crf.put(DIM_DATA_WRITER, new CrfSuiteAdapter().getDataWriterClass().getName());
        crf.put(DIM_FEATURE_USE_SPARSE, new CrfSuiteAdapter().useSparseFeatures());

        Dimension<Map<String, Object>> dimClassifiers = Dimension.createBundle("classification", crf);
		
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
		
		/* Reader for Wall-Street-Journal Corpus - uncomment for reading the WSJ */
//		CollectionReaderDescription trainReader = createReaderDescription(
//				PennTreebankChunkedReader.class,
//				PennTreebankChunkedReader.PARAM_LANGUAGE, "en", 
//				PennTreebankChunkedReader.PARAM_SOURCE_LOCATION, train,
//				PennTreebankChunkedReader.PARAM_PATTERNS, "**/*.pos");

//		CollectionReaderDescription testReader = createReaderDescription(
//				PennTreebankChunkedReader.class,
//				PennTreebankChunkedReader.PARAM_LANGUAGE, "en", 
//				PennTreebankChunkedReader.PARAM_SOURCE_LOCATION, test,
//				PennTreebankChunkedReader.PARAM_PATTERNS, "**/*.pos");
		
        Map<String, Object> dimReaders = new HashMap<String, Object>();
		dimReaders.put(DIM_READER_TRAIN, trainReader);
		dimReaders.put(DIM_READER_TEST, testReader);

		Dimension<TcFeatureSet> dimFeatureSets = Dimension.create(DIM_FEATURE_SET,
				new TcFeatureSet(
						TcFeatureFactory.create(TargetSurfaceFormContextFeature.class,
												TargetSurfaceFormContextFeature.PARAM_RELATIVE_TARGET_ANNOTATION_INDEX, -2),
						TcFeatureFactory.create(TargetSurfaceFormContextFeature.class,
												TargetSurfaceFormContextFeature.PARAM_RELATIVE_TARGET_ANNOTATION_INDEX, -1),
						TcFeatureFactory.create(TargetSurfaceFormContextFeature.class,
												TargetSurfaceFormContextFeature.PARAM_RELATIVE_TARGET_ANNOTATION_INDEX, 0),
						TcFeatureFactory.create(BrownCluster.class,
												BrownCluster.PARAM_LOWER_CASE, true,
												BrownCluster.PARAM_RESOURCE_LOCATION, brown, 
												BrownCluster.PARAM_NORMALIZATION, false)

		));

		ParameterSpace pSpace;
		pSpace = new ParameterSpace(Dimension.createBundle("readers", dimReaders),
				Dimension.create(DIM_LEARNING_MODE, LM_SINGLE_LABEL), Dimension.create(DIM_FEATURE_MODE, FM_SEQUENCE),
				dimFeatureSets, dimClassifiers);

		return pSpace;
	}

	// ##### TRAIN-TEST #####
	protected void runTrainTest(ParameterSpace pSpace) throws Exception {

		ExperimentTrainTest experiment = new ExperimentTrainTest("CrfExperiment");
		experiment.setPreprocessing(createEngineDescription(SequenceOutcomeAnnotator.class));
		experiment.setParameterSpace(pSpace);
		experiment.setExecutionPolicy(ExecutionPolicy.RUN_AGAIN);
		experiment.addReport(BatchRuntimeReport.class);

		// Run
		Lab.getInstance().run(experiment);
	}
}
