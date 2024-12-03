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
    sphere { m*<0.939491276498047,0.5682615338848293,0.4213571468120439>, 1 }        
    sphere {  m*<1.1831833970206125,0.6155412593128876,3.411067346009233>, 1 }
    sphere {  m*<3.676430586083149,0.6155412593128874,-0.8062148624813859>, 1 }
    sphere {  m*<-2.6420270218781043,6.123968111228155,-1.6962633445278021>, 1}
    sphere { m*<-3.8361613794316773,-7.751258562091822,-2.401644798979544>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1831833970206125,0.6155412593128876,3.411067346009233>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5 }
    cylinder { m*<3.676430586083149,0.6155412593128874,-0.8062148624813859>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5}
    cylinder { m*<-2.6420270218781043,6.123968111228155,-1.6962633445278021>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5 }
    cylinder {  m*<-3.8361613794316773,-7.751258562091822,-2.401644798979544>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5}

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
    sphere { m*<0.939491276498047,0.5682615338848293,0.4213571468120439>, 1 }        
    sphere {  m*<1.1831833970206125,0.6155412593128876,3.411067346009233>, 1 }
    sphere {  m*<3.676430586083149,0.6155412593128874,-0.8062148624813859>, 1 }
    sphere {  m*<-2.6420270218781043,6.123968111228155,-1.6962633445278021>, 1}
    sphere { m*<-3.8361613794316773,-7.751258562091822,-2.401644798979544>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1831833970206125,0.6155412593128876,3.411067346009233>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5 }
    cylinder { m*<3.676430586083149,0.6155412593128874,-0.8062148624813859>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5}
    cylinder { m*<-2.6420270218781043,6.123968111228155,-1.6962633445278021>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5 }
    cylinder {  m*<-3.8361613794316773,-7.751258562091822,-2.401644798979544>, <0.939491276498047,0.5682615338848293,0.4213571468120439>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    