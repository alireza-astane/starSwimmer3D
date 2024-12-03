#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 1 }        
    sphere {  m*<0.8558489493811244,0.24260088329712337,9.338860286721506>, 1 }
    sphere {  m*<8.223636147703928,-0.04249136749513793,-5.231817142352424>, 1 }
    sphere {  m*<-6.672327045985066,6.4805900061255,-3.7410102391708167>, 1}
    sphere { m*<-3.2809340524490676,-6.665773970534013,-1.7689237158747684>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8558489493811244,0.24260088329712337,9.338860286721506>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5 }
    cylinder { m*<8.223636147703928,-0.04249136749513793,-5.231817142352424>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5}
    cylinder { m*<-6.672327045985066,6.4805900061255,-3.7410102391708167>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5 }
    cylinder {  m*<-3.2809340524490676,-6.665773970534013,-1.7689237158747684>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 1 }        
    sphere {  m*<0.8558489493811244,0.24260088329712337,9.338860286721506>, 1 }
    sphere {  m*<8.223636147703928,-0.04249136749513793,-5.231817142352424>, 1 }
    sphere {  m*<-6.672327045985066,6.4805900061255,-3.7410102391708167>, 1}
    sphere { m*<-3.2809340524490676,-6.665773970534013,-1.7689237158747684>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8558489493811244,0.24260088329712337,9.338860286721506>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5 }
    cylinder { m*<8.223636147703928,-0.04249136749513793,-5.231817142352424>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5}
    cylinder { m*<-6.672327045985066,6.4805900061255,-3.7410102391708167>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5 }
    cylinder {  m*<-3.2809340524490676,-6.665773970534013,-1.7689237158747684>, <-0.5633185448190374,-0.7473380305827938,-0.5104298103136418>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    