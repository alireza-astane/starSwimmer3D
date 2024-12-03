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
    sphere { m*<0.3233189534987409,0.8190665298209497,0.0592785041853755>, 1 }        
    sphere {  m*<0.5640540582404325,0.9477766080012753,3.0468332753059264>, 1 }
    sphere {  m*<3.0580273475049973,0.9211005052073242,-1.169931021265807>, 1 }
    sphere {  m*<-1.2982964063941496,3.1475404742395527,-0.9146672612305932>, 1}
    sphere { m*<-3.3902316313806833,-6.200868438652714,-2.0923292701496843>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5640540582404325,0.9477766080012753,3.0468332753059264>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5 }
    cylinder { m*<3.0580273475049973,0.9211005052073242,-1.169931021265807>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5}
    cylinder { m*<-1.2982964063941496,3.1475404742395527,-0.9146672612305932>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5 }
    cylinder {  m*<-3.3902316313806833,-6.200868438652714,-2.0923292701496843>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5}

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
    sphere { m*<0.3233189534987409,0.8190665298209497,0.0592785041853755>, 1 }        
    sphere {  m*<0.5640540582404325,0.9477766080012753,3.0468332753059264>, 1 }
    sphere {  m*<3.0580273475049973,0.9211005052073242,-1.169931021265807>, 1 }
    sphere {  m*<-1.2982964063941496,3.1475404742395527,-0.9146672612305932>, 1}
    sphere { m*<-3.3902316313806833,-6.200868438652714,-2.0923292701496843>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5640540582404325,0.9477766080012753,3.0468332753059264>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5 }
    cylinder { m*<3.0580273475049973,0.9211005052073242,-1.169931021265807>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5}
    cylinder { m*<-1.2982964063941496,3.1475404742395527,-0.9146672612305932>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5 }
    cylinder {  m*<-3.3902316313806833,-6.200868438652714,-2.0923292701496843>, <0.3233189534987409,0.8190665298209497,0.0592785041853755>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    