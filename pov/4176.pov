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
    sphere { m*<-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 1 }        
    sphere {  m*<0.17369940686197416,0.09765134957894994,3.7999319317888265>, 1 }
    sphere {  m*<2.5682490978682213,0.017817984496524678,-1.6506924852556921>, 1 }
    sphere {  m*<-1.7880746560309257,2.2442579535287495,-1.3954287252204787>, 1}
    sphere { m*<-1.520287434993094,-2.643433988875148,-1.205882440057906>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17369940686197416,0.09765134957894994,3.7999319317888265>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5 }
    cylinder { m*<2.5682490978682213,0.017817984496524678,-1.6506924852556921>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5}
    cylinder { m*<-1.7880746560309257,2.2442579535287495,-1.3954287252204787>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5 }
    cylinder {  m*<-1.520287434993094,-2.643433988875148,-1.205882440057906>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5}

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
    sphere { m*<-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 1 }        
    sphere {  m*<0.17369940686197416,0.09765134957894994,3.7999319317888265>, 1 }
    sphere {  m*<2.5682490978682213,0.017817984496524678,-1.6506924852556921>, 1 }
    sphere {  m*<-1.7880746560309257,2.2442579535287495,-1.3954287252204787>, 1}
    sphere { m*<-1.520287434993094,-2.643433988875148,-1.205882440057906>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17369940686197416,0.09765134957894994,3.7999319317888265>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5 }
    cylinder { m*<2.5682490978682213,0.017817984496524678,-1.6506924852556921>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5}
    cylinder { m*<-1.7880746560309257,2.2442579535287495,-1.3954287252204787>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5 }
    cylinder {  m*<-1.520287434993094,-2.643433988875148,-1.205882440057906>, <-0.1664592961380357,-0.08421599088984945,-0.4214829598045087>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    