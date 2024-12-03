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
    sphere { m*<-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 1 }        
    sphere {  m*<0.21986097525418546,0.2841119565406267,8.555887834744423>, 1 }
    sphere {  m*<5.17559718052759,0.05348116943235151,-4.415910627921136>, 1 }
    sphere {  m*<-2.623051436624592,2.1644359586768207,-2.2785190105294326>, 1}
    sphere { m*<-2.3552642155867605,-2.7232559837270767,-2.0889727253668617>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21986097525418546,0.2841119565406267,8.555887834744423>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5 }
    cylinder { m*<5.17559718052759,0.05348116943235151,-4.415910627921136>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5}
    cylinder { m*<-2.623051436624592,2.1644359586768207,-2.2785190105294326>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5 }
    cylinder {  m*<-2.3552642155867605,-2.7232559837270767,-2.0889727253668617>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5}

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
    sphere { m*<-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 1 }        
    sphere {  m*<0.21986097525418546,0.2841119565406267,8.555887834744423>, 1 }
    sphere {  m*<5.17559718052759,0.05348116943235151,-4.415910627921136>, 1 }
    sphere {  m*<-2.623051436624592,2.1644359586768207,-2.2785190105294326>, 1}
    sphere { m*<-2.3552642155867605,-2.7232559837270767,-2.0889727253668617>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21986097525418546,0.2841119565406267,8.555887834744423>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5 }
    cylinder { m*<5.17559718052759,0.05348116943235151,-4.415910627921136>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5}
    cylinder { m*<-2.623051436624592,2.1644359586768207,-2.2785190105294326>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5 }
    cylinder {  m*<-2.3552642155867605,-2.7232559837270767,-2.0889727253668617>, <-0.9683266242836319,-0.1644909074680012,-1.3631080998741214>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    