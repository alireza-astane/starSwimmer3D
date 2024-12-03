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
    sphere { m*<0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 1 }        
    sphere {  m*<1.0839585405973282,1.2261339125770073e-18,3.8116377164905053>, 1 }
    sphere {  m*<5.679848538648348,5.548174994716113e-18,-1.1485799796256273>, 1 }
    sphere {  m*<-3.9533780142998145,8.164965809277259,-2.2677390095203647>, 1}
    sphere { m*<-3.9533780142998145,-8.164965809277259,-2.2677390095203673>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0839585405973282,1.2261339125770073e-18,3.8116377164905053>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5 }
    cylinder { m*<5.679848538648348,5.548174994716113e-18,-1.1485799796256273>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5}
    cylinder { m*<-3.9533780142998145,8.164965809277259,-2.2677390095203647>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5 }
    cylinder {  m*<-3.9533780142998145,-8.164965809277259,-2.2677390095203673>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5}

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
    sphere { m*<0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 1 }        
    sphere {  m*<1.0839585405973282,1.2261339125770073e-18,3.8116377164905053>, 1 }
    sphere {  m*<5.679848538648348,5.548174994716113e-18,-1.1485799796256273>, 1 }
    sphere {  m*<-3.9533780142998145,8.164965809277259,-2.2677390095203647>, 1}
    sphere { m*<-3.9533780142998145,-8.164965809277259,-2.2677390095203673>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0839585405973282,1.2261339125770073e-18,3.8116377164905053>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5 }
    cylinder { m*<5.679848538648348,5.548174994716113e-18,-1.1485799796256273>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5}
    cylinder { m*<-3.9533780142998145,8.164965809277259,-2.2677390095203647>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5 }
    cylinder {  m*<-3.9533780142998145,-8.164965809277259,-2.2677390095203673>, <0.9277977217302714,-1.5605190089517843e-18,0.8156988736679478>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    