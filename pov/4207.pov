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
    sphere { m*<-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 1 }        
    sphere {  m*<0.18790408980156637,0.1052459455297291,3.976213942399487>, 1 }
    sphere {  m*<2.565040608231498,0.01610255145892693,-1.6905102697560856>, 1 }
    sphere {  m*<-1.7912831456676495,2.2425425204911518,-1.4352465097208722>, 1}
    sphere { m*<-1.5234959246298176,-2.6451494219127456,-1.2457002245582995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18790408980156637,0.1052459455297291,3.976213942399487>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5 }
    cylinder { m*<2.565040608231498,0.01610255145892693,-1.6905102697560856>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5}
    cylinder { m*<-1.7912831456676495,2.2425425204911518,-1.4352465097208722>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5 }
    cylinder {  m*<-1.5234959246298176,-2.6451494219127456,-1.2457002245582995>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5}

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
    sphere { m*<-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 1 }        
    sphere {  m*<0.18790408980156637,0.1052459455297291,3.976213942399487>, 1 }
    sphere {  m*<2.565040608231498,0.01610255145892693,-1.6905102697560856>, 1 }
    sphere {  m*<-1.7912831456676495,2.2425425204911518,-1.4352465097208722>, 1}
    sphere { m*<-1.5234959246298176,-2.6451494219127456,-1.2457002245582995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18790408980156637,0.1052459455297291,3.976213942399487>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5 }
    cylinder { m*<2.565040608231498,0.01610255145892693,-1.6905102697560856>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5}
    cylinder { m*<-1.7912831456676495,2.2425425204911518,-1.4352465097208722>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5 }
    cylinder {  m*<-1.5234959246298176,-2.6451494219127456,-1.2457002245582995>, <-0.16966778577475927,-0.0859314239274472,-0.4613007443049022>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    