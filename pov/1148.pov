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
    sphere { m*<0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 1 }        
    sphere {  m*<0.23589785078720438,-1.2810128852750065e-18,4.121251313691179>, 1 }
    sphere {  m*<8.624817689082596,4.545243359247145e-18,-1.942434704340467>, 1 }
    sphere {  m*<-4.53660687764156,8.164965809277259,-2.1681499194814515>, 1}
    sphere { m*<-4.53660687764156,-8.164965809277259,-2.168149919481455>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23589785078720438,-1.2810128852750065e-18,4.121251313691179>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5 }
    cylinder { m*<8.624817689082596,4.545243359247145e-18,-1.942434704340467>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5}
    cylinder { m*<-4.53660687764156,8.164965809277259,-2.1681499194814515>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5 }
    cylinder {  m*<-4.53660687764156,-8.164965809277259,-2.168149919481455>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5}

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
    sphere { m*<0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 1 }        
    sphere {  m*<0.23589785078720438,-1.2810128852750065e-18,4.121251313691179>, 1 }
    sphere {  m*<8.624817689082596,4.545243359247145e-18,-1.942434704340467>, 1 }
    sphere {  m*<-4.53660687764156,8.164965809277259,-2.1681499194814515>, 1}
    sphere { m*<-4.53660687764156,-8.164965809277259,-2.168149919481455>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23589785078720438,-1.2810128852750065e-18,4.121251313691179>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5 }
    cylinder { m*<8.624817689082596,4.545243359247145e-18,-1.942434704340467>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5}
    cylinder { m*<-4.53660687764156,8.164965809277259,-2.1681499194814515>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5 }
    cylinder {  m*<-4.53660687764156,-8.164965809277259,-2.168149919481455>, <0.20811135974081282,-2.9192140286257985e-18,1.121379133510163>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    