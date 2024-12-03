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
    sphere { m*<-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 1 }        
    sphere {  m*<0.19636612020334732,-0.02749420803251995,9.020731179381848>, 1 }
    sphere {  m*<7.551717558203321,-0.11641448402687672,-5.5587621106634995>, 1 }
    sphere {  m*<-5.047463782230442,4.074194361618995,-2.795445700847254>, 1}
    sphere { m*<-2.5481980147071526,-3.3449386079926353,-1.4891738983475669>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19636612020334732,-0.02749420803251995,9.020731179381848>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5 }
    cylinder { m*<7.551717558203321,-0.11641448402687672,-5.5587621106634995>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5}
    cylinder { m*<-5.047463782230442,4.074194361618995,-2.795445700847254>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5 }
    cylinder {  m*<-2.5481980147071526,-3.3449386079926353,-1.4891738983475669>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5}

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
    sphere { m*<-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 1 }        
    sphere {  m*<0.19636612020334732,-0.02749420803251995,9.020731179381848>, 1 }
    sphere {  m*<7.551717558203321,-0.11641448402687672,-5.5587621106634995>, 1 }
    sphere {  m*<-5.047463782230442,4.074194361618995,-2.795445700847254>, 1}
    sphere { m*<-2.5481980147071526,-3.3449386079926353,-1.4891738983475669>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19636612020334732,-0.02749420803251995,9.020731179381848>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5 }
    cylinder { m*<7.551717558203321,-0.11641448402687672,-5.5587621106634995>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5}
    cylinder { m*<-5.047463782230442,4.074194361618995,-2.795445700847254>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5 }
    cylinder {  m*<-2.5481980147071526,-3.3449386079926353,-1.4891738983475669>, <-1.2497773914703167,-0.7173770699426671,-0.8501493929497465>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    