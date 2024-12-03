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
    sphere { m*<1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 1 }        
    sphere {  m*<1.2886664100466394,1.2202470057263984e-18,3.7230521952393065>, 1 }
    sphere {  m*<4.931675847454603,5.8938160320178605e-18,-0.9160747630237169>, 1 }
    sphere {  m*<-3.827432706679229,8.164965809277259,-2.290198791482311>, 1}
    sphere { m*<-3.827432706679229,-8.164965809277259,-2.2901987914823136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2886664100466394,1.2202470057263984e-18,3.7230521952393065>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5 }
    cylinder { m*<4.931675847454603,5.8938160320178605e-18,-0.9160747630237169>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5}
    cylinder { m*<-3.827432706679229,8.164965809277259,-2.290198791482311>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5 }
    cylinder {  m*<-3.827432706679229,-8.164965809277259,-2.2901987914823136>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5}

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
    sphere { m*<1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 1 }        
    sphere {  m*<1.2886664100466394,1.2202470057263984e-18,3.7230521952393065>, 1 }
    sphere {  m*<4.931675847454603,5.8938160320178605e-18,-0.9160747630237169>, 1 }
    sphere {  m*<-3.827432706679229,8.164965809277259,-2.290198791482311>, 1}
    sphere { m*<-3.827432706679229,-8.164965809277259,-2.2901987914823136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2886664100466394,1.2202470057263984e-18,3.7230521952393065>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5 }
    cylinder { m*<4.931675847454603,5.8938160320178605e-18,-0.9160747630237169>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5}
    cylinder { m*<-3.827432706679229,8.164965809277259,-2.290198791482311>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5 }
    cylinder {  m*<-3.827432706679229,-8.164965809277259,-2.2901987914823136>, <1.0934951978962582,-5.471412192842688e-19,0.7293997408933486>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    