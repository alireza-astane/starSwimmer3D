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
    sphere { m*<0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 1 }        
    sphere {  m*<0.6901968743398541,-1.7520546084360743e-18,3.965133104083762>, 1 }
    sphere {  m*<7.064552986567047,2.123770858267713e-18,-1.5409670012182624>, 1 }
    sphere {  m*<-4.212569374069241,8.164965809277259,-2.223271106457908>, 1}
    sphere { m*<-4.212569374069241,-8.164965809277259,-2.2232711064579105>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6901968743398541,-1.7520546084360743e-18,3.965133104083762>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5 }
    cylinder { m*<7.064552986567047,2.123770858267713e-18,-1.5409670012182624>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5}
    cylinder { m*<-4.212569374069241,8.164965809277259,-2.223271106457908>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5 }
    cylinder {  m*<-4.212569374069241,-8.164965809277259,-2.2232711064579105>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5}

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
    sphere { m*<0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 1 }        
    sphere {  m*<0.6901968743398541,-1.7520546084360743e-18,3.965133104083762>, 1 }
    sphere {  m*<7.064552986567047,2.123770858267713e-18,-1.5409670012182624>, 1 }
    sphere {  m*<-4.212569374069241,8.164965809277259,-2.223271106457908>, 1}
    sphere { m*<-4.212569374069241,-8.164965809277259,-2.2232711064579105>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6901968743398541,-1.7520546084360743e-18,3.965133104083762>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5 }
    cylinder { m*<7.064552986567047,2.123770858267713e-18,-1.5409670012182624>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5}
    cylinder { m*<-4.212569374069241,8.164965809277259,-2.223271106457908>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5 }
    cylinder {  m*<-4.212569374069241,-8.164965809277259,-2.2232711064579105>, <0.5997908784068352,-5.205417100849447e-18,0.9664924856333297>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    