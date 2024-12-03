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
    sphere { m*<1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 1 }        
    sphere {  m*<1.4967656400139417,7.312982706087953e-20,3.6254788623093157>, 1 }
    sphere {  m*<4.126127963278905,6.2426569455810376e-18,-0.6439742980580414>, 1 }
    sphere {  m*<-3.70572138434776,8.164965809277259,-2.313197283739057>, 1}
    sphere { m*<-3.70572138434776,-8.164965809277259,-2.3131972837390595>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4967656400139417,7.312982706087953e-20,3.6254788623093157>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5 }
    cylinder { m*<4.126127963278905,6.2426569455810376e-18,-0.6439742980580414>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5}
    cylinder { m*<-3.70572138434776,8.164965809277259,-2.313197283739057>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5 }
    cylinder {  m*<-3.70572138434776,-8.164965809277259,-2.3131972837390595>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5}

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
    sphere { m*<1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 1 }        
    sphere {  m*<1.4967656400139417,7.312982706087953e-20,3.6254788623093157>, 1 }
    sphere {  m*<4.126127963278905,6.2426569455810376e-18,-0.6439742980580414>, 1 }
    sphere {  m*<-3.70572138434776,8.164965809277259,-2.313197283739057>, 1}
    sphere { m*<-3.70572138434776,-8.164965809277259,-2.3131972837390595>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4967656400139417,7.312982706087953e-20,3.6254788623093157>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5 }
    cylinder { m*<4.126127963278905,6.2426569455810376e-18,-0.6439742980580414>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5}
    cylinder { m*<-3.70572138434776,8.164965809277259,-2.313197283739057>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5 }
    cylinder {  m*<-3.70572138434776,-8.164965809277259,-2.3131972837390595>, <1.2583428061095503,-6.233022510088557e-19,0.6349581587669196>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    