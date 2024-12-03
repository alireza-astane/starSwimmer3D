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
    sphere { m*<0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 1 }        
    sphere {  m*<1.0537226013759475,2.2000242108439067e-18,3.8241634502652833>, 1 }
    sphere {  m*<5.78814677424542,6.0057176080330645e-18,-1.1808851095872221>, 1 }
    sphere {  m*<-3.9725034880372774,8.164965809277259,-2.264401392855657>, 1}
    sphere { m*<-3.9725034880372774,-8.164965809277259,-2.2644013928556594>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0537226013759475,2.2000242108439067e-18,3.8241634502652833>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5 }
    cylinder { m*<5.78814677424542,6.0057176080330645e-18,-1.1808851095872221>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5}
    cylinder { m*<-3.9725034880372774,8.164965809277259,-2.264401392855657>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5 }
    cylinder {  m*<-3.9725034880372774,-8.164965809277259,-2.2644013928556594>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5}

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
    sphere { m*<0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 1 }        
    sphere {  m*<1.0537226013759475,2.2000242108439067e-18,3.8241634502652833>, 1 }
    sphere {  m*<5.78814677424542,6.0057176080330645e-18,-1.1808851095872221>, 1 }
    sphere {  m*<-3.9725034880372774,8.164965809277259,-2.264401392855657>, 1}
    sphere { m*<-3.9725034880372774,-8.164965809277259,-2.2644013928556594>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0537226013759475,2.2000242108439067e-18,3.8241634502652833>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5 }
    cylinder { m*<5.78814677424542,6.0057176080330645e-18,-1.1808851095872221>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5}
    cylinder { m*<-3.9725034880372774,8.164965809277259,-2.264401392855657>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5 }
    cylinder {  m*<-3.9725034880372774,-8.164965809277259,-2.2644013928556594>, <0.9030329874686511,-1.4253164350446415e-18,0.8279446847361744>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    