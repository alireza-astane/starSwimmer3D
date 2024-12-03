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
    sphere { m*<0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 1 }        
    sphere {  m*<1.1515251251607086,8.453459275834628e-19,3.783145956989264>, 1 }
    sphere {  m*<5.436073294706774,5.071412129891931e-18,-1.0746790491428537>, 1 }
    sphere {  m*<-3.911123157804693,8.164965809277259,-2.2751704264377013>, 1}
    sphere { m*<-3.911123157804693,-8.164965809277259,-2.275170426437705>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1515251251607086,8.453459275834628e-19,3.783145956989264>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5 }
    cylinder { m*<5.436073294706774,5.071412129891931e-18,-1.0746790491428537>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5}
    cylinder { m*<-3.911123157804693,8.164965809277259,-2.2751704264377013>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5 }
    cylinder {  m*<-3.911123157804693,-8.164965809277259,-2.275170426437705>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5}

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
    sphere { m*<0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 1 }        
    sphere {  m*<1.1515251251607086,8.453459275834628e-19,3.783145956989264>, 1 }
    sphere {  m*<5.436073294706774,5.071412129891931e-18,-1.0746790491428537>, 1 }
    sphere {  m*<-3.911123157804693,8.164965809277259,-2.2751704264377013>, 1}
    sphere { m*<-3.911123157804693,-8.164965809277259,-2.275170426437705>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1515251251607086,8.453459275834628e-19,3.783145956989264>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5 }
    cylinder { m*<5.436073294706774,5.071412129891931e-18,-1.0746790491428537>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5}
    cylinder { m*<-3.911123157804693,8.164965809277259,-2.2751704264377013>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5 }
    cylinder {  m*<-3.911123157804693,-8.164965809277259,-2.275170426437705>, <0.9828708134423033,-1.689388223543376e-18,0.7878838727423512>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    