export interface DiamondInput {
	carat: number;
}

export interface DiamondOutput {
	message: string;
	data: {
		carat: number;
	};
	df: string;
	preds: number;
}
