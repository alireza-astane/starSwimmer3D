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
    sphere { m*<-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 1 }        
    sphere {  m*<0.026812024788821498,0.11379614959802786,8.934324037878627>, 1 }
    sphere {  m*<7.382163462788793,0.024875873603670717,-5.645169252166729>, 1 }
    sphere {  m*<-4.173984322956577,3.158599628253742,-2.349161191506764>, 1}
    sphere { m*<-2.7777292589133995,-3.051043814849343,-1.6067733719201258>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.026812024788821498,0.11379614959802786,8.934324037878627>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5 }
    cylinder { m*<7.382163462788793,0.024875873603670717,-5.645169252166729>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5}
    cylinder { m*<-4.173984322956577,3.158599628253742,-2.349161191506764>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5 }
    cylinder {  m*<-2.7777292589133995,-3.051043814849343,-1.6067733719201258>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5}

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
    sphere { m*<-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 1 }        
    sphere {  m*<0.026812024788821498,0.11379614959802786,8.934324037878627>, 1 }
    sphere {  m*<7.382163462788793,0.024875873603670717,-5.645169252166729>, 1 }
    sphere {  m*<-4.173984322956577,3.158599628253742,-2.349161191506764>, 1}
    sphere { m*<-2.7777292589133995,-3.051043814849343,-1.6067733719201258>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.026812024788821498,0.11379614959802786,8.934324037878627>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5 }
    cylinder { m*<7.382163462788793,0.024875873603670717,-5.645169252166729>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5}
    cylinder { m*<-4.173984322956577,3.158599628253742,-2.349161191506764>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5 }
    cylinder {  m*<-2.7777292589133995,-3.051043814849343,-1.6067733719201258>, <-1.4300766206916085,-0.454665982820342,-0.9427196837621572>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    