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
    sphere { m*<-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 1 }        
    sphere {  m*<0.18224044240285436,0.28332989937443254,8.588869959474103>, 1 }
    sphere {  m*<5.428150768364626,0.06132439790282268,-4.5702760783990275>, 1 }
    sphere {  m*<-2.700063377883893,2.161890374429514,-2.2354278445224693>, 1}
    sphere { m*<-2.432276156846062,-2.725801567974383,-2.045881559359899>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18224044240285436,0.28332989937443254,8.588869959474103>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5 }
    cylinder { m*<5.428150768364626,0.06132439790282268,-4.5702760783990275>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5}
    cylinder { m*<-2.700063377883893,2.161890374429514,-2.2354278445224693>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5 }
    cylinder {  m*<-2.432276156846062,-2.725801567974383,-2.045881559359899>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5}

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
    sphere { m*<-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 1 }        
    sphere {  m*<0.18224044240285436,0.28332989937443254,8.588869959474103>, 1 }
    sphere {  m*<5.428150768364626,0.06132439790282268,-4.5702760783990275>, 1 }
    sphere {  m*<-2.700063377883893,2.161890374429514,-2.2354278445224693>, 1}
    sphere { m*<-2.432276156846062,-2.725801567974383,-2.045881559359899>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18224044240285436,0.28332989937443254,8.588869959474103>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5 }
    cylinder { m*<5.428150768364626,0.06132439790282268,-4.5702760783990275>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5}
    cylinder { m*<-2.700063377883893,2.161890374429514,-2.2354278445224693>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5 }
    cylinder {  m*<-2.432276156846062,-2.725801567974383,-2.045881559359899>, <-1.0423266421292192,-0.16708828089697914,-1.3256155292094016>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    