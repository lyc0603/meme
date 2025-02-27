const solanaWeb3 = require('@solana/web3.js');
const searchAddress = '8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj'
const endpoint = 'https://api.mainnet-beta.solana.com';
const solanaConnection = new solanaWeb3.Connection(endpoint);

const getTransactions = async(address, numTx) => {
    const pubKey = new solanaWeb3.PublicKey(address);
    let transactionList = await solanaConnection.getSignaturesForAddress(pubKey, {limit:numTx});
    
    //Add this code
    let signatureList = transactionList.map(transaction=>transaction.signature);
    let transactionDetails = await solanaConnection.getParsedTransactions(signatureList, {maxSupportedTransactionVersion:0});
    //--END of new code 

    transactionList.forEach((transaction, i) => {
        const date = new Date(transaction.blockTime*1000);
        console.log(`Transaction No: ${i+1}`);
        console.log(`Signature: ${transaction.signature}`);
        console.log(`Time: ${date}`);
        console.log(`Status: ${transaction.confirmationStatus}`);
        console.log(("-").repeat(20));
    })
}

getTransactions(searchAddress, 1);