import { expect } from 'chai';
import { describe } from 'mocha';
import { FaceClassifier } from './FaceClassifier';
import { DbHelper } from '../../../pmr-server/src/dbclient'

describe.only('FaceClassifier', () => {
    let db: DbHelper;

    it('.getInstance()', async () => {
        let classifier = await FaceClassifier.getInstance();
        expect(classifier).to.instanceOf(FaceClassifier);
    });
    it('.predict()', async () => {
        let classifier = await FaceClassifier.getInstance();
        db = new DbHelper();
        let face = await db.getFaceByName('Vladimir Putin', 2);
        await classifier.predict(Float64Array.from(face[0].embedding)); // lazy loading
        let pred = await classifier.predict(Float64Array.from(face[0].embedding));
        expect(pred.label).to.equal('Vladimir Putin');
        console.log(`Predicted: ${pred.label} (${pred.confidences[pred.label] * 100}%), Actual Vladimir Putin`);
    });
});