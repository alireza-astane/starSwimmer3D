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
    sphere { m*<0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 1 }        
    sphere {  m*<0.2517025088932234,-9.42501249420933e-19,4.116120807596759>, 1 }
    sphere {  m*<8.57087758435972,4.91891140533769e-18,-1.9290870480863676>, 1 }
    sphere {  m*<-4.524929236770974,8.164965809277259,-2.170146853439194>, 1}
    sphere { m*<-4.524929236770974,-8.164965809277259,-2.1701468534391974>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2517025088932234,-9.42501249420933e-19,4.116120807596759>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5 }
    cylinder { m*<8.57087758435972,4.91891140533769e-18,-1.9290870480863676>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5}
    cylinder { m*<-4.524929236770974,8.164965809277259,-2.170146853439194>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5 }
    cylinder {  m*<-4.524929236770974,-8.164965809277259,-2.1701468534391974>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5}

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
    sphere { m*<0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 1 }        
    sphere {  m*<0.2517025088932234,-9.42501249420933e-19,4.116120807596759>, 1 }
    sphere {  m*<8.57087758435972,4.91891140533769e-18,-1.9290870480863676>, 1 }
    sphere {  m*<-4.524929236770974,8.164965809277259,-2.170146853439194>, 1}
    sphere { m*<-4.524929236770974,-8.164965809277259,-2.1701468534391974>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2517025088932234,-9.42501249420933e-19,4.116120807596759>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5 }
    cylinder { m*<8.57087758435972,4.91891140533769e-18,-1.9290870480863676>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5}
    cylinder { m*<-4.524929236770974,8.164965809277259,-2.170146853439194>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5 }
    cylinder {  m*<-4.524929236770974,-8.164965809277259,-2.1701468534391974>, <0.22194710774602708,-2.5859111783576114e-18,1.1162674461290896>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    