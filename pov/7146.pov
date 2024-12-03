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
    sphere { m*<-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 1 }        
    sphere {  m*<0.6783157650941728,-0.14403172733814174,9.256646879478026>, 1 }
    sphere {  m*<8.04610296341697,-0.42912397813040437,-5.314030549595906>, 1 }
    sphere {  m*<-6.84986023027202,6.093957395490253,-3.8232236464143012>, 1}
    sphere { m*<-2.391578833016948,-4.728931956168075,-1.3570743867876063>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6783157650941728,-0.14403172733814174,9.256646879478026>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5 }
    cylinder { m*<8.04610296341697,-0.42912397813040437,-5.314030549595906>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5}
    cylinder { m*<-6.84986023027202,6.093957395490253,-3.8232236464143012>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5 }
    cylinder {  m*<-2.391578833016948,-4.728931956168075,-1.3570743867876063>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5}

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
    sphere { m*<-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 1 }        
    sphere {  m*<0.6783157650941728,-0.14403172733814174,9.256646879478026>, 1 }
    sphere {  m*<8.04610296341697,-0.42912397813040437,-5.314030549595906>, 1 }
    sphere {  m*<-6.84986023027202,6.093957395490253,-3.8232236464143012>, 1}
    sphere { m*<-2.391578833016948,-4.728931956168075,-1.3570743867876063>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6783157650941728,-0.14403172733814174,9.256646879478026>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5 }
    cylinder { m*<8.04610296341697,-0.42912397813040437,-5.314030549595906>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5}
    cylinder { m*<-6.84986023027202,6.093957395490253,-3.8232236464143012>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5 }
    cylinder {  m*<-2.391578833016948,-4.728931956168075,-1.3570743867876063>, <-0.7408517291059894,-1.1339706412180592,-0.5926432175571237>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    